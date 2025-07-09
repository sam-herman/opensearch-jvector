/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;

import java.io.IOException;
import java.time.Clock;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL;

@Log4j2
public class JVectorWriter extends KnnVectorsWriter {
    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.class);
    private static final FlatVectorsFormat FLAT_VECTORS_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );
    private final List<FieldWriter<?>> fields = new ArrayList<>();

    private final IndexOutput meta;
    private final IndexOutput vectorIndex;
    private final FlatVectorsWriter flatVectorWriter;
    private final String indexDataFileName;
    private final String baseDataFileName;
    private final SegmentWriteState segmentWriteState;
    private final int maxConn;
    private final int beamWidth;
    private final float degreeOverflow;
    private final float alpha;
    private final Function<Integer, Integer> numberOfSubspacesPerVectorSupplier; // Number of subspaces used per vector for PQ quantization
                                                                                 // as a function of the original dimension
    private final int minimumBatchSizeForQuantization; // Threshold for the vector count above which we will trigger PQ quantization
    private final boolean mergeOnDisk;

    private boolean finished = false;

    public JVectorWriter(
        SegmentWriteState segmentWriteState,
        int maxConn,
        int beamWidth,
        float degreeOverflow,
        float alpha,
        Function<Integer, Integer> numberOfSubspacesPerVectorSupplier,
        int minimumBatchSizeForQuantization,
        boolean mergeOnDisk
    ) throws IOException {
        this.segmentWriteState = segmentWriteState;
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.degreeOverflow = degreeOverflow;
        this.alpha = alpha;
        this.numberOfSubspacesPerVectorSupplier = numberOfSubspacesPerVectorSupplier;
        this.minimumBatchSizeForQuantization = minimumBatchSizeForQuantization;
        this.mergeOnDisk = mergeOnDisk;
        this.flatVectorWriter = FLAT_VECTORS_FORMAT.fieldsWriter(segmentWriteState);
        String metaFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            JVectorFormat.META_EXTENSION
        );

        this.indexDataFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            JVectorFormat.VECTOR_INDEX_EXTENSION
        );
        this.baseDataFileName = segmentWriteState.segmentInfo.name + "_" + segmentWriteState.segmentSuffix;

        boolean success = false;
        try {
            meta = segmentWriteState.directory.createOutput(metaFileName, segmentWriteState.context);
            vectorIndex = segmentWriteState.directory.createOutput(indexDataFileName, segmentWriteState.context);
            CodecUtil.writeIndexHeader(
                meta,
                JVectorFormat.META_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );

            CodecUtil.writeIndexHeader(
                vectorIndex,
                JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );

            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        log.info("Adding field {} in segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            final String errorMessage = "byte[] vectors are not supported in JVector. "
                + "Instead you should only use float vectors and leverage product quantization during indexing."
                + "This can provides much greater savings in storage and memory";
            log.error(errorMessage);
            throw new UnsupportedOperationException(errorMessage);
        }
        final FlatFieldVectorsWriter<?> flatFieldVectorsWriter = flatVectorWriter.addField(fieldInfo);
        FieldWriter<?> newField = new FieldWriter<>(fieldInfo, segmentWriteState.segmentInfo.name, flatFieldVectorsWriter, maxConn, beamWidth, degreeOverflow, alpha);

        fields.add(newField);
        return newField;
    }

    public KnnFieldVectorsWriter<?> addMergeField(FieldInfo fieldInfo, FloatVectorValues mergeFloatVector, RandomAccessVectorValues ravv)
        throws IOException {
        log.info("Adding merge field {} in segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            final String errorMessage = "byte[] vectors are not supported in JVector. "
                + "Instead you should only use float vectors and leverage product quantization during indexing."
                + "This can provides much greater savings in storage and memory";
            log.error(errorMessage);
            throw new UnsupportedOperationException(errorMessage);
        }
        return new FieldWriter<>(fieldInfo, segmentWriteState.segmentInfo.name, mergeFloatVector, ravv, maxConn, beamWidth, degreeOverflow, alpha);
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        log.info("Merging field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        CloseableRandomVectorScorerSupplier scorerSupplier = flatVectorWriter.mergeOneFieldToIndex(fieldInfo, mergeState);
        var success = false;
        try {
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    var byteWriter = (FieldWriter<byte[]>) addField(fieldInfo);
                    ByteVectorValues mergedBytes = MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
                    var iterator = mergedBytes.iterator();
                    for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                        byteWriter.addValue(doc, mergedBytes.vectorValue(doc));
                    }
                    writeField(byteWriter);
                    break;
                case FLOAT32:
                    final FieldWriter<float[]> floatVectorFieldWriter;
                    FloatVectorValues mergeFloatVector = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
                    if (mergeOnDisk) {
                        final var ravv = new RandomAccessMergedFloatVectorValues(fieldInfo, mergeState, scorerSupplier);
                        floatVectorFieldWriter = (FieldWriter<float[]>) addMergeField(fieldInfo, mergeFloatVector, ravv);
                    } else {
                        floatVectorFieldWriter = (FieldWriter<float[]>) addField(fieldInfo);
                        var itr = mergeFloatVector.iterator();
                        for (int doc = itr.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = itr.nextDoc()) {
                            floatVectorFieldWriter.addValue(doc, mergeFloatVector.vectorValue(doc));
                        }
                    }
                    writeField(floatVectorFieldWriter);
                    break;
            }
            success = true;
            log.info("Completed Merge field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        } finally {
            IOUtils.close(scorerSupplier);
            if (success) {
                // IOUtils.close(scorerSupplier);
            } else {
                // IOUtils.closeWhileHandlingException(scorerSupplier);
            }
        }
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        log.info("Flushing {} fields", fields.size());

        log.info("Flushing flat vectors");
        flatVectorWriter.flush(maxDoc, sortMap);
        log.info("Flushing jVector graph index");
        for (FieldWriter<?> field : fields) {
            if (sortMap == null) {
                writeField(field);
            } else {
                throw new UnsupportedOperationException("Not implemented yet");
                // writeSortingField(field, sortMap);
            }
        }
    }

    private void writeField(FieldWriter<?> fieldData) throws IOException {
        log.info(
            "Writing field {} with vector count: {}, for segment: {}",
            fieldData.fieldInfo.name,
            fieldData.randomAccessVectorValues.size(),
            segmentWriteState.segmentInfo.name
        );
        final PQVectors pqVectors;
        final BuildScoreProvider buildScoreProvider;
        if (fieldData.randomAccessVectorValues.size() >= minimumBatchSizeForQuantization) {
            log.info("Calculating codebooks and compressed vectors for field {}", fieldData.fieldInfo.name);
            pqVectors = getPQVectors(fieldData);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(getVectorSimilarityFunction(fieldData.fieldInfo), pqVectors);
        } else {
            log.info(
                "Vector count: {}, less than limit to trigger PQ quantization: {}, for field {}, will use full precision vectors instead.",
                fieldData.randomAccessVectorValues.size(),
                minimumBatchSizeForQuantization,
                fieldData.fieldInfo.name
            );
            pqVectors = null;
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                fieldData.randomAccessVectorValues,
                getVectorSimilarityFunction(fieldData.fieldInfo)
            );
        }

        OnHeapGraphIndex graph = fieldData.getGraph(buildScoreProvider);
        final var vectorIndexFieldMetadata = writeGraph(graph, fieldData, pqVectors);
        meta.writeInt(fieldData.fieldInfo.number);
        vectorIndexFieldMetadata.toOutput(meta);
    }

    /**
     * Writes the graph and PQ codebooks and compressed vectors to the vector index file
     * @param graph graph
     * @param fieldData fieldData
     * @return Tuple of start offset and length of the graph
     * @throws IOException IOException
     */
    private VectorIndexFieldMetadata writeGraph(OnHeapGraphIndex graph, FieldWriter<?> fieldData, PQVectors pqVectors) throws IOException {
        // field data file, which contains the graph
        final String vectorIndexFieldFileName = baseDataFileName
            + "_"
            + fieldData.fieldInfo.name
            + "."
            + JVectorFormat.VECTOR_INDEX_EXTENSION;

        try (
            IndexOutput indexOutput = segmentWriteState.directory.createOutput(vectorIndexFieldFileName, segmentWriteState.context);
            final var jVectorIndexWriter = new JVectorIndexWriter(indexOutput)
        ) {
            // Header for the field data file
            CodecUtil.writeIndexHeader(
                indexOutput,
                JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );
            final long startOffset = indexOutput.getFilePointer();

            log.info("Writing graph to {}", vectorIndexFieldFileName);
            var resultBuilder = VectorIndexFieldMetadata.builder()
                .fieldNumber(fieldData.fieldInfo.number)
                .vectorEncoding(fieldData.fieldInfo.getVectorEncoding())
                .vectorSimilarityFunction(fieldData.fieldInfo.getVectorSimilarityFunction())
                .vectorDimension(fieldData.randomAccessVectorValues.dimension());

            try (
                var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, jVectorIndexWriter).with(
                    new InlineVectors(fieldData.randomAccessVectorValues.dimension())
                ).build()
            ) {
                var suppliers = Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(fieldData.randomAccessVectorValues.getVector(nodeId))
                );
                writer.write(suppliers);
                long endGraphOffset = jVectorIndexWriter.position();
                resultBuilder.vectorIndexOffset(startOffset);
                resultBuilder.vectorIndexLength(endGraphOffset - startOffset);

                // If PQ is enabled and we have enough vectors, write the PQ codebooks and compressed vectors
                if (pqVectors != null) {
                    log.info(
                        "Writing PQ codebooks and vectors for field {} since the size is {} >= {}",
                        fieldData.fieldInfo.name,
                        fieldData.randomAccessVectorValues.size(),
                        minimumBatchSizeForQuantization
                    );
                    resultBuilder.pqCodebooksAndVectorsOffset(endGraphOffset);
                    // write the compressed vectors and codebooks to disk
                    pqVectors.write(jVectorIndexWriter);
                    resultBuilder.pqCodebooksAndVectorsLength(jVectorIndexWriter.position() - endGraphOffset);
                } else {
                    resultBuilder.pqCodebooksAndVectorsOffset(0);
                    resultBuilder.pqCodebooksAndVectorsLength(0);
                }
                CodecUtil.writeFooter(indexOutput);
            }

            return resultBuilder.build();
        }
    }

    private PQVectors getPQVectors(FieldWriter<?> fieldData) throws IOException {
        log.info("Computing PQ codebooks for field {} for {} vectors", fieldData.fieldInfo.name, fieldData.randomAccessVectorValues.size());
        final long start = Clock.systemDefaultZone().millis();
        final var M = numberOfSubspacesPerVectorSupplier.apply(fieldData.randomAccessVectorValues.dimension());
        final var numberOfClustersPerSubspace = Math.min(256, fieldData.randomAccessVectorValues.size()); // number of centroids per
        // subspace
        ProductQuantization pq = ProductQuantization.compute(
            fieldData.randomAccessVectorValues,
            M, // number of subspaces
            numberOfClustersPerSubspace, // number of centroids per subspace
            fieldData.fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN, // center the dataset
            UNWEIGHTED,
            SIMD_POOL,
            ForkJoinPool.commonPool()
        );

        final long end = Clock.systemDefaultZone().millis();
        log.info("Computed PQ codebooks for field {}, in {} millis", fieldData.fieldInfo.name, end - start);
        log.info(
            "Encoding and building PQ vectors for field {} for {} vectors",
            fieldData.fieldInfo.name,
            fieldData.randomAccessVectorValues.size()
        );
        PQVectors pqVectors = (PQVectors) pq.encodeAll(fieldData.randomAccessVectorValues);
        log.info(
            "Encoded and built PQ vectors for field {}, original size: {} bytes, compressed size: {} bytes",
            fieldData.fieldInfo.name,
            pqVectors.getOriginalSize(),
            pqVectors.getCompressedSize()
        );
        return pqVectors;
    }

    @Value
    @Builder(toBuilder = true)
    @AllArgsConstructor
    public static class VectorIndexFieldMetadata {
        int fieldNumber;
        VectorEncoding vectorEncoding;
        VectorSimilarityFunction vectorSimilarityFunction;
        int vectorDimension;
        long vectorIndexOffset;
        long vectorIndexLength;
        long pqCodebooksAndVectorsOffset;
        long pqCodebooksAndVectorsLength;

        public void toOutput(IndexOutput out) throws IOException {
            out.writeInt(fieldNumber);
            out.writeInt(vectorEncoding.ordinal());
            out.writeInt(JVectorReader.VectorSimilarityMapper.distFuncToOrd(vectorSimilarityFunction));
            out.writeVInt(vectorDimension);
            out.writeVLong(vectorIndexOffset);
            out.writeVLong(vectorIndexLength);
            out.writeVLong(pqCodebooksAndVectorsOffset);
            out.writeVLong(pqCodebooksAndVectorsLength);
        }

        public VectorIndexFieldMetadata(IndexInput in) throws IOException {
            this.fieldNumber = in.readInt();
            this.vectorEncoding = readVectorEncoding(in);
            this.vectorSimilarityFunction = JVectorReader.VectorSimilarityMapper.ordToLuceneDistFunc(in.readInt());
            this.vectorDimension = in.readVInt();
            this.vectorIndexOffset = in.readVLong();
            this.vectorIndexLength = in.readVLong();
            this.pqCodebooksAndVectorsOffset = in.readVLong();
            this.pqCodebooksAndVectorsLength = in.readVLong();
        }

    }

    @Override
    public void finish() throws IOException {
        log.info("Finishing segment {}", segmentWriteState.segmentInfo.name);
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;

        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }

        if (vectorIndex != null) {
            CodecUtil.writeFooter(vectorIndex);
        }

        flatVectorWriter.finish();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorIndex, flatVectorWriter);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldWriter<?> field : fields) {
            // the field tracks the delegate field usage
            total += field.ramBytesUsed();
        }
        return total;
    }

    /**
     * The FieldWriter class is responsible for writing vector field data into index segments.
     * It provides functionality to process vector values as those being added, manage memory usage, and build HNSW graph
     * indexing structures for efficient retrieval during search queries.
     *
     * @param <T> The type of vector value to be handled by the writer.
     * This is often specialized to support specific implementations, such as float[] or byte[] vectors.
     */
    public static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
        @Getter
        private final FieldInfo fieldInfo;
        private int lastDocID = -1;
        private final String segmentName;
        @Getter
        private final RandomAccessVectorValues randomAccessVectorValues;
        private final FloatVectorValues mergedFloatVector;
        private final FlatFieldVectorsWriter<T> flatFieldVectorsWriter;
        private final int maxConn;
        private final int beamWidth;
        private final float degreeOverflow;
        private final float alpha;

        // For merge fields only
        FieldWriter(FieldInfo fieldInfo, String segmentName, FloatVectorValues mergedFloatVector, RandomAccessVectorValues ravv, int maxConn, int beamWidth, float degreeOverflow, float alpha) {
            this.maxConn = maxConn;
            this.beamWidth = beamWidth;
            this.degreeOverflow = degreeOverflow;
            this.alpha = alpha;
            this.flatFieldVectorsWriter = null;
            this.randomAccessVectorValues = ravv;
            this.mergedFloatVector = mergedFloatVector;
            // This unmodifiable list makes sure that the addition of values outside what's already in ravv will fail.
            this.fieldInfo = fieldInfo;
            this.segmentName = segmentName;
        }

        FieldWriter(FieldInfo fieldInfo, String segmentName, FlatFieldVectorsWriter<T> flatFieldVectorsWriter, int maxConn, int beamWidth, float degreeOverflow, float alpha) {
            /**
             * For creating a new field from a flat field vectors writer.
             */
            this.flatFieldVectorsWriter = flatFieldVectorsWriter;
            this.maxConn = maxConn;
            this.beamWidth = beamWidth;
            this.degreeOverflow = degreeOverflow;
            this.alpha = alpha;
            this.randomAccessVectorValues = new RandomAccessVectorValuesOverFlatFields(flatFieldVectorsWriter, fieldInfo);
            this.mergedFloatVector = null;
            this.fieldInfo = fieldInfo;
            this.segmentName = segmentName;
        }

        @Override
        public void addValue(int docID, T vectorValue) throws IOException {
            log.trace("Adding value {} to field {} in segment {}", vectorValue, fieldInfo.name, segmentName);
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                    "VectorValuesField \""
                        + fieldInfo.name
                        + "\" appears more than once in this document (only one value is allowed per field)"
                );
            }
            if (vectorValue instanceof float[]) {
                flatFieldVectorsWriter.addValue(docID, vectorValue);
            } else if (vectorValue instanceof byte[]) {
                final String errorMessage = "byte[] vectors are not supported in JVector. "
                    + "Instead you should only use float vectors and leverage product quantization during indexing."
                    + "This can provides much greater savings in storage and memory";
                log.error(errorMessage);
                throw new UnsupportedOperationException(errorMessage);
            } else {
                throw new IllegalArgumentException("Unsupported vector type: " + vectorValue.getClass());
            }

            lastDocID = docID;
        }

        @Override
        public T copyValue(T vectorValue) {
            throw new UnsupportedOperationException("copyValue not supported");
        }

        @Override
        public long ramBytesUsed() {
            return SHALLOW_SIZE + flatFieldVectorsWriter.ramBytesUsed();
        }

        /**
         * This method will return the graph index for the field
         * @return OnHeapGraphIndex
         * @throws IOException IOException
         */
        public OnHeapGraphIndex getGraph(BuildScoreProvider buildScoreProvider) throws IOException {
            final GraphIndexBuilder graphIndexBuilder = new GraphIndexBuilder(
                buildScoreProvider,
                fieldInfo.getVectorDimension(),
                maxConn,
                beamWidth,
                degreeOverflow,
                alpha,
                true
            );

            /*
             * We cannot always use randomAccessVectorValues for the graph building
             * because it's size will not always correspond to the document count.
             * To have the right mapping from docId to vector ordinal we need to use the mergedFloatVector.
             * This is the case when we are merging segments and we might have more documents than vectors.
             */
            final long start = Clock.systemDefaultZone().millis();
            final OnHeapGraphIndex graphIndex;
            var vv = randomAccessVectorValues.threadLocalSupplier();
            if (mergedFloatVector != null) {
                log.info("Building graph from merged float vector");
                var itr = mergedFloatVector.iterator();
                // Gather a list of valid document Ids to be streamed later for parallel graph construction
                final List<Integer> docIds = new ArrayList<>();
                int doc;
                while ((doc = itr.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
                    docIds.add(doc);
                }

                // parallel graph construction from the merge documents Ids
                SIMD_POOL.submit(
                    () -> docIds.stream().parallel().forEach(node -> { graphIndexBuilder.addGraphNode(node, vv.get().getVector(node)); })
                ).join();
            } else {
                log.info("Building graph from random access vector values");
                int size = randomAccessVectorValues.size();

                SIMD_POOL.submit(() -> {
                    IntStream.range(0, size)
                        .parallel()
                        .forEach(node -> { graphIndexBuilder.addGraphNode(node, vv.get().getVector(node)); });
                }).join();
            }
            graphIndexBuilder.cleanup();
            graphIndex = graphIndexBuilder.getGraph();
            final long end = Clock.systemDefaultZone().millis();

            log.info("Built graph for field {} in segment {} in {} millis", fieldInfo.name, segmentName, end - start);
            return graphIndex;
        }
    }

    static io.github.jbellis.jvector.vector.VectorSimilarityFunction getVectorSimilarityFunction(FieldInfo fieldInfo) {
        log.info("Matching vector similarity function {} for field {}", fieldInfo.getVectorSimilarityFunction(), fieldInfo.name);
        return switch (fieldInfo.getVectorSimilarityFunction()) {
            case EUCLIDEAN -> io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
            case COSINE -> io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
            case DOT_PRODUCT -> io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
            default -> throw new IllegalArgumentException("Unsupported similarity function: " + fieldInfo.getVectorSimilarityFunction());
        };
    }

    /**
     * Implementation of RandomAccessVectorValues that directly uses the source
     * FloatVectorValues from multiple segments without copying the vectors.
     */
    static class RandomAccessMergedFloatVectorValues implements RandomAccessVectorValues {
        private static final int READER_ID = 0;
        private static final int READER_ORD = 1;

        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

        // Array of sub-readers
        private final KnnVectorsReader[] readers;
        private final FloatVectorValues[] perReaderFloatVectorValues;

        // For each ordinal, stores which reader and which ordinal in that reader
        private final int[][] ordMapping;

        // Total number of vectors
        private final int size;
        // Total number of documents including those without values
        private final int totalDocsCount;

        // Vector dimension
        private final int dimension;
        private final CloseableRandomVectorScorerSupplier scorerSupplier;

        private String fieldName;

        /**
         * Creates a random access view over merged float vector values.
         *
         * @param fieldInfo Field info for the vector field
         * @param mergeState Merge state containing readers and doc maps
         */
        public RandomAccessMergedFloatVectorValues(
            FieldInfo fieldInfo,
            MergeState mergeState,
            CloseableRandomVectorScorerSupplier scorerSupplier
        ) throws IOException {
            this.fieldName = fieldInfo.name;
            this.scorerSupplier = scorerSupplier;
            this.totalDocsCount = Math.toIntExact(Arrays.stream(mergeState.maxDocs).asLongStream().sum());
            // Count total vectors and collect readers
            int totalVectorsCount = 0;
            int dimension = 0;
            // We explicitly show that the readers are of JVectorFloatVectorValues that are capable of random access
            List<KnnVectorsReader> allReaders = new ArrayList<>();

            for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
                FieldInfos fieldInfos = mergeState.fieldInfos[i];
                if (KnnVectorsWriter.MergedVectorValues.hasVectorValues(fieldInfos, fieldName)) {
                    KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
                    if (reader != null) {
                        FloatVectorValues values = reader.getFloatVectorValues(fieldName);
                        if (values != null) {
                            allReaders.add(reader);
                            totalVectorsCount += values.size();
                            dimension = Math.max(dimension, values.dimension());
                        }
                    }
                }
            }

            assert (totalVectorsCount <= totalDocsCount) : "Total number of vectors exceeds the total number of documents";
            assert (dimension > 0) : "No vectors found for field " + fieldName;

            this.size = totalVectorsCount;
            this.readers = new KnnVectorsReader[allReaders.size()];
            for (int i = 0; i < readers.length; i++) {
                readers[i] = allReaders.get(i);
            }
            this.perReaderFloatVectorValues = new FloatVectorValues[readers.length];
            this.dimension = dimension;

            // Build mapping from global ordinal to [readerIndex, readerOrd]
            this.ordMapping = new int[totalDocsCount][2];

            int documentsIterated = 0;

            // Simulate the merge process to build the ordinal mapping
            // This is similar to what DocIDMerger would do but tracks ordinals
            MergeState.DocMap[] docMaps = mergeState.docMaps;
            // For each reader
            for (int readerIdx = 0; readerIdx < readers.length; readerIdx++) {
                final FloatVectorValues values = readers[readerIdx].getFloatVectorValues(fieldName);
                perReaderFloatVectorValues[readerIdx] = values;
                // For each vector in this reader
                KnnVectorValues.DocIndexIterator it = values.iterator();
                for (int docId = it.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = it.nextDoc()) {
                    if (docMaps[readerIdx].get(docId) == -1) {
                        log.warn(
                            "Document {} in reader {} is not mapped to a global ordinal from the merge docMaps. Will skip this document for now",
                            docId,
                            readerIdx
                        );
                    } else {
                        // Mapping from global ordinal to [readerIndex, readerOrd]
                        final int globalOrd = docMaps[readerIdx].get(docId);
                        ordMapping[globalOrd][READER_ID] = readerIdx; // Reader index
                        ordMapping[globalOrd][READER_ORD] = docId; // Ordinal in reader
                    }

                    documentsIterated++;
                }
            }

            if (documentsIterated < totalVectorsCount) {
                throw new IllegalStateException(
                    "More documents were expected than what was found in the readers."
                        + "Expected at least number of total vectors: "
                        + totalVectorsCount
                        + " but found only: "
                        + documentsIterated
                        + " documents."
                );
            }

            log.debug("Created RandomAccessMergedFloatVectorValues with {} total vectors from {} readers", size, readers.length);
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int ord) {
            if (ord < 0 || ord >= totalDocsCount) {
                throw new IllegalArgumentException("Ordinal out of bounds: " + ord);
            }

            try {

                final int readerIdx = ordMapping[ord][READER_ID];
                final int readerOrd = ordMapping[ord][READER_ORD];

                // Access to float values is not thread safe
                synchronized (this) {
                    final FloatVectorValues values = perReaderFloatVectorValues[readerIdx];
                    final float[] vector = values.vectorValue(readerOrd);
                    final float[] copy = new float[vector.length];
                    System.arraycopy(vector, 0, copy, 0, vector.length);
                    return VECTOR_TYPE_SUPPORT.createFloatVector(copy);
                }
            } catch (IOException e) {
                log.error("Error retrieving vector at ordinal {}", ord, e);
                throw new RuntimeException(e);
            }
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException("Copy not supported");
        }
    }

    static class RandomAccessVectorValuesOverFlatFields implements RandomAccessVectorValues {
        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

        private final FlatFieldVectorsWriter<?> flatFieldVectorsWriter;
        private final int dimension;

        RandomAccessVectorValuesOverFlatFields(FlatFieldVectorsWriter<?> flatFieldVectorsWriter, FieldInfo fieldInfo) {
            this.flatFieldVectorsWriter = flatFieldVectorsWriter;
            this.dimension = fieldInfo.getVectorDimension();
        }

        @Override
        public int size() {
            return flatFieldVectorsWriter.getVectors().size();
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            final float[] vector = (float[]) flatFieldVectorsWriter.getVectors().get(nodeId);
            return VECTOR_TYPE_SUPPORT.createFloatVector(vector);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException("Copy not supported");
        }
    }
}
