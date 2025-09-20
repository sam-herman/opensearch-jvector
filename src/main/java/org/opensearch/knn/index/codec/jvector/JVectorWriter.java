/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
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
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.time.Clock;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL;

/**
 * JVectorWriter is responsible for writing vector data into index segments using the JVector library.
 *
 * <h2>Persisting the JVector Graph Index</h2>
 *
 * <p>
 * Flushing data into disk segments occurs in two scenarios:
 * <ol>
 *     <li>When the segment is being flushed to disk (e.g., when a new segment is created) via {@link #flush(int, Sorter.DocMap)}</li>
 *     <li>When the segment is a result of a merge (e.g., when multiple segments are merged into one) via {@link #mergeOneField(FieldInfo, MergeState)}</li>
 * </ol>
 *
 * <h2>jVector Graph Ordinal to Lucene Document ID Mapping</h2>
 *
 * <p>
 * JVector keeps its own ordinals to identify its nodes. Those ordinals can be different from the Lucene document IDs.
 * Document IDs in Lucene can change after a merge operation. Therefore, we need to maintain a mapping between
 * JVector ordinals and Lucene document IDs that can hold across merges.
 * <p>
 * Document IDs in Lucene are mapped across merges and sorts using the {@link org.apache.lucene.index.MergeState.DocMap} for merges and {@link org.apache.lucene.index.Sorter.DocMap} for flush/sorts.
 * For jVector however, we don't want to modify the ordinals in the jVector graph, and therefore we need to maintain a mapping between the jVector ordinals and the new Lucene document IDs.
 * This is achieved by keeping checkpoints of the {@link JVectorLuceneDocMap} class in the index metadata and allowing us to update the mapping as needed across merges by constructing a new mapping from the previous mapping and the {@link MergeState.DocMap} provided in the {@link MergeState}.
 * And across sorts with {@link JVectorLuceneDocMap#update(Sorter.DocMap)} during flushes.
 * <p>
 *
 */
@Log4j2
public class JVectorWriter extends KnnVectorsWriter {
    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.class);
    private final List<FieldWriter<?>> fields = new ArrayList<>();

    private final IndexOutput meta;
    private final IndexOutput vectorIndex;
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

    private boolean finished = false;

    public JVectorWriter(
        SegmentWriteState segmentWriteState,
        int maxConn,
        int beamWidth,
        float degreeOverflow,
        float alpha,
        Function<Integer, Integer> numberOfSubspacesPerVectorSupplier,
        int minimumBatchSizeForQuantization
    ) throws IOException {
        this.segmentWriteState = segmentWriteState;
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.degreeOverflow = degreeOverflow;
        this.alpha = alpha;
        this.numberOfSubspacesPerVectorSupplier = numberOfSubspacesPerVectorSupplier;
        this.minimumBatchSizeForQuantization = minimumBatchSizeForQuantization;
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
        FieldWriter<?> newField = new FieldWriter<>(fieldInfo, segmentWriteState.segmentInfo.name);

        fields.add(newField);
        return newField;
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        log.info("Merging field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        var success = false;
        try {
            final long mergeStart = Clock.systemDefaultZone().millis();
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    throw new UnsupportedEncodingException("Byte vectors are not supported in JVector.");
                case FLOAT32:
                    final var mergeRavv = new RandomAccessMergedFloatVectorValues(fieldInfo, mergeState);
                    mergeRavv.merge();
                    break;
            }
            final long mergeEnd = Clock.systemDefaultZone().millis();
            final long mergeTime = mergeEnd - mergeStart;
            KNNCounter.KNN_GRAPH_MERGE_TIME.add(mergeTime);
            success = true;
            log.info("Completed Merge field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        } catch (Exception e) {
            log.error("Error merging field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name, e);
            throw e;
        }
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        log.info("Flushing {} fields", fields.size());

        log.info("Flushing jVector graph index");
        for (FieldWriter<?> field : fields) {
            final RandomAccessVectorValues randomAccessVectorValues = field.randomAccessVectorValues;
            final int[] newToOldOrds = new int[randomAccessVectorValues.size()];
            for (int ord = 0; ord < randomAccessVectorValues.size(); ord++) {
                newToOldOrds[ord] = ord;
            }
            final PQVectors pqVectors;
            final FieldInfo fieldInfo = field.fieldInfo;
            if (randomAccessVectorValues.size() >= minimumBatchSizeForQuantization) {
                log.info("Calculating codebooks and compressed vectors for field {}", fieldInfo.name);
                pqVectors = getPQVectors(newToOldOrds, randomAccessVectorValues, fieldInfo);
            } else {
                log.info(
                    "Vector count: {}, less than limit to trigger PQ quantization: {}, for field {}, will use full precision vectors instead.",
                    randomAccessVectorValues.size(),
                    minimumBatchSizeForQuantization,
                    fieldInfo.name
                );
                pqVectors = null;
            }

            // Generate the ord to doc mapping
            final int[] ordinalsToDocIds = new int[randomAccessVectorValues.size()];
            for (int ord = 0; ord < randomAccessVectorValues.size(); ord++) {
                ordinalsToDocIds[ord] = field.docIds.get(ord);
            }
            final JVectorLuceneDocMap jVectorLuceneDocMap = new JVectorLuceneDocMap(ordinalsToDocIds);
            if (sortMap != null) {
                jVectorLuceneDocMap.update(sortMap);
            }

            writeField(field.fieldInfo, field.randomAccessVectorValues, pqVectors, newToOldOrds, jVectorLuceneDocMap);

        }
    }

    private void writeField(
        FieldInfo fieldInfo,
        RandomAccessVectorValues randomAccessVectorValues,
        PQVectors pqVectors,
        int[] newToOldOrds,
        JVectorLuceneDocMap jVectorLuceneDocMap
    ) throws IOException {
        log.info(
            "Writing field {} with vector count: {}, for segment: {}",
            fieldInfo.name,
            randomAccessVectorValues.size(),
            segmentWriteState.segmentInfo.name
        );
        final BuildScoreProvider buildScoreProvider;
        if (pqVectors == null) {
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                randomAccessVectorValues,
                getVectorSimilarityFunction(fieldInfo)
            );
        } else {
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(getVectorSimilarityFunction(fieldInfo), pqVectors);
        }

        OnHeapGraphIndex graph = getGraph(
            buildScoreProvider,
            randomAccessVectorValues,
            newToOldOrds,
            fieldInfo,
            segmentWriteState.segmentInfo.name
        );
        final var vectorIndexFieldMetadata = writeGraph(
            graph,
            randomAccessVectorValues,
            fieldInfo,
            pqVectors,
            newToOldOrds,
            jVectorLuceneDocMap
        );
        meta.writeInt(fieldInfo.number);
        vectorIndexFieldMetadata.toOutput(meta);

        log.info("Writing neighbors score cache for field {}", fieldInfo.name);
        NeighborsScoreCache neighborsScoreCache = new NeighborsScoreCache(graph);
        // field data file, which contains the graph
        final String neighborsScoreCacheIndexFieldFileName = baseDataFileName
            + "_"
            + fieldInfo.name
            + "."
            + JVectorFormat.NEIGHBORS_SCORE_CACHE_EXTENSION;
        try (
            IndexOutput indexOutput = segmentWriteState.directory.createOutput(
                neighborsScoreCacheIndexFieldFileName,
                segmentWriteState.context
            );
            final var jVectorIndexWriter = new JVectorIndexWriter(indexOutput)
        ) {
            CodecUtil.writeIndexHeader(
                indexOutput,
                JVectorFormat.NEIGHBORS_SCORE_CACHE_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );
            neighborsScoreCache.write(jVectorIndexWriter);
            CodecUtil.writeFooter(indexOutput);
        }
    }

    /**
     * Writes the graph and PQ codebooks and compressed vectors to the vector index file
     * @param graph graph
     * @param randomAccessVectorValues random access vector values
     * @param fieldInfo field info
     * @return Tuple of start offset and length of the graph
     * @throws IOException IOException
     */
    private VectorIndexFieldMetadata writeGraph(
        OnHeapGraphIndex graph,
        RandomAccessVectorValues randomAccessVectorValues,
        FieldInfo fieldInfo,
        PQVectors pqVectors,
        int[] newToOldOrds,
        JVectorLuceneDocMap jVectorLuceneDocMap
    ) throws IOException {
        // field data file, which contains the graph
        final String vectorIndexFieldFileName = baseDataFileName + "_" + fieldInfo.name + "." + JVectorFormat.VECTOR_INDEX_EXTENSION;

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
                .fieldNumber(fieldInfo.number)
                .vectorEncoding(fieldInfo.getVectorEncoding())
                .vectorSimilarityFunction(fieldInfo.getVectorSimilarityFunction())
                .vectorDimension(randomAccessVectorValues.dimension())
                .jVectorLuceneDocMap(jVectorLuceneDocMap);

            try (
                var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, jVectorIndexWriter).with(
                    new InlineVectors(randomAccessVectorValues.dimension())
                ).build()
            ) {
                var suppliers = Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(randomAccessVectorValues.getVector(newToOldOrds[nodeId]))
                );
                writer.write(suppliers);
                long endGraphOffset = jVectorIndexWriter.position();
                resultBuilder.vectorIndexOffset(startOffset);
                resultBuilder.vectorIndexLength(endGraphOffset - startOffset);

                // If PQ is enabled and we have enough vectors, write the PQ codebooks and compressed vectors
                if (pqVectors != null) {
                    log.info(
                        "Writing PQ codebooks and vectors for field {} since the size is {} >= {}",
                        fieldInfo.name,
                        randomAccessVectorValues.size(),
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

    private PQVectors getPQVectors(int[] newToOldOrds, RandomAccessVectorValues randomAccessVectorValues, FieldInfo fieldInfo)
        throws IOException {
        final String fieldName = fieldInfo.name;
        final VectorSimilarityFunction vectorSimilarityFunction = fieldInfo.getVectorSimilarityFunction();
        log.info("Computing PQ codebooks for field {} for {} vectors", fieldName, randomAccessVectorValues.size());
        final long start = Clock.systemDefaultZone().millis();
        final var M = numberOfSubspacesPerVectorSupplier.apply(randomAccessVectorValues.dimension());
        final var numberOfClustersPerSubspace = Math.min(256, randomAccessVectorValues.size()); // number of centroids per
        // subspace
        ProductQuantization pq = ProductQuantization.compute(
            randomAccessVectorValues,
            M, // number of subspaces
            numberOfClustersPerSubspace, // number of centroids per subspace
            vectorSimilarityFunction == VectorSimilarityFunction.EUCLIDEAN, // center the dataset
            UNWEIGHTED,
            SIMD_POOL,
            ForkJoinPool.commonPool()
        );

        final long end = Clock.systemDefaultZone().millis();
        final long trainingTime = end - start;
        log.info("Computed PQ codebooks for field {}, in {} millis", fieldName, trainingTime);
        KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(trainingTime);
        log.info("Encoding and building PQ vectors for field {} for {} vectors", fieldName, randomAccessVectorValues.size());
        // PQVectors pqVectors = pq.encodeAll(randomAccessVectorValues, SIMD_POOL);
        PQVectors pqVectors = PQVectors.encodeAndBuild(pq, newToOldOrds.length, newToOldOrds, randomAccessVectorValues, SIMD_POOL);
        log.info(
            "Encoded and built PQ vectors for field {}, original size: {} bytes, compressed size: {} bytes",
            fieldName,
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
        JVectorLuceneDocMap jVectorLuceneDocMap;

        public void toOutput(IndexOutput out) throws IOException {
            out.writeInt(fieldNumber);
            out.writeInt(vectorEncoding.ordinal());
            out.writeInt(JVectorReader.VectorSimilarityMapper.distFuncToOrd(vectorSimilarityFunction));
            out.writeVInt(vectorDimension);
            out.writeVLong(vectorIndexOffset);
            out.writeVLong(vectorIndexLength);
            out.writeVLong(pqCodebooksAndVectorsOffset);
            out.writeVLong(pqCodebooksAndVectorsLength);
            jVectorLuceneDocMap.toOutput(out);
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
            this.jVectorLuceneDocMap = new JVectorLuceneDocMap(in);
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

    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorIndex);
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
    static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
        @Getter
        private final FieldInfo fieldInfo;
        private int lastDocID = -1;
        private final String segmentName;
        private final RandomAccessVectorValues randomAccessVectorValues;
        // The ordering of docIds matches the ordering of vectors, the index in this list corresponds to the jVector ordinal
        private final List<VectorFloat<?>> vectors = new ArrayList<>();
        private final List<Integer> docIds = new ArrayList<>();

        FieldWriter(FieldInfo fieldInfo, String segmentName) {
            /**
             * For creating a new field from a flat field vectors writer.
             */
            this.randomAccessVectorValues = new ListRandomAccessVectorValues(vectors, fieldInfo.getVectorDimension());
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
            docIds.add(docID);
            if (vectorValue instanceof float[]) {
                vectors.add(VECTOR_TYPE_SUPPORT.createFloatVector(vectorValue));
            } else if (vectorValue instanceof byte[]) {
                final String errorMessage = "byte[] vectors are not supported in JVector. "
                    + "Instead you should only use float vectors and leverage product quantization during indexing."
                    + "This can provides much greater savings in storage and memory";
                log.error("{}", errorMessage);
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
            return SHALLOW_SIZE + (long) vectors.size() * fieldInfo.getVectorDimension() * Float.BYTES;
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
    class RandomAccessMergedFloatVectorValues implements RandomAccessVectorValues {
        private static final int READER_ID = 0;
        private static final int READER_ORD = 1;
        private static final int LEADING_READER_IDX = 0;

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
        private final FieldInfo fieldInfo;
        private final MergeState mergeState;
        private final JVectorLuceneDocMap jVectorLuceneDocMap;
        private final int[] newToOldOrds;

        /**
         * Creates a random access view over merged float vector values.
         *
         * @param fieldInfo Field info for the vector field
         * @param mergeState Merge state containing readers and doc maps
         */
        public RandomAccessMergedFloatVectorValues(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
            this.totalDocsCount = Math.toIntExact(Arrays.stream(mergeState.maxDocs).asLongStream().sum());
            this.fieldInfo = fieldInfo;
            this.mergeState = mergeState;

            final String fieldName = fieldInfo.name;

            // Count total vectors, collect readers and identify leading reader, collect base ordinals to later be used to build the mapping
            // between global ordinals and global lucene doc ids
            int totalVectorsCount = 0;
            int totalLiveVectorsCount = 0;
            int dimension = 0;
            int tempLeadingReaderIdx = -1;
            int vectorsCountInLeadingReader = -1;
            List<KnnVectorsReader> allReaders = new ArrayList<>();
            final MergeState.DocMap[] docMaps = mergeState.docMaps.clone();
            final Bits[] liveDocs = mergeState.liveDocs.clone();
            final int[] baseOrds = new int[mergeState.knnVectorsReaders.length];
            final int[] deletedOrds = new int[mergeState.knnVectorsReaders.length]; // counts the number of deleted documents in each reader
                                                                                    // that previously had a vector
            final int[] newBaseOrds = new int[mergeState.knnVectorsReaders.length]; // the new base ordinals after taking into account the
                                                                                    // deleted documents
            for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
                FieldInfos fieldInfos = mergeState.fieldInfos[i];
                baseOrds[i] = totalVectorsCount;
                newBaseOrds[i] = totalLiveVectorsCount;
                if (MergedVectorValues.hasVectorValues(fieldInfos, fieldName)) {
                    KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
                    if (reader != null) {
                        FloatVectorValues values = reader.getFloatVectorValues(fieldName);
                        if (values != null) {
                            allReaders.add(reader);
                            int vectorCountInReader = values.size();
                            int liveVectorCountInReader = 0;
                            KnnVectorValues.DocIndexIterator it = values.iterator();
                            while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                                if (liveDocs[i] == null || liveDocs[i].get(it.docID())) {
                                    liveVectorCountInReader++;
                                } else {
                                    deletedOrds[i]++;
                                }
                            }
                            if (liveVectorCountInReader >= vectorsCountInLeadingReader) {
                                vectorsCountInLeadingReader = liveVectorCountInReader;
                                tempLeadingReaderIdx = i;
                            }
                            totalVectorsCount += vectorCountInReader;
                            totalLiveVectorsCount += liveVectorCountInReader;
                            dimension = Math.max(dimension, values.dimension());
                        }
                    }
                }
            }

            assert (totalVectorsCount <= totalDocsCount) : "Total number of vectors exceeds the total number of documents";
            assert (totalLiveVectorsCount <= totalVectorsCount) : "Total number of live vectors exceeds the total number of vectors";
            assert (dimension > 0) : "No vectors found for field " + fieldName;

            this.size = totalVectorsCount;
            this.readers = new KnnVectorsReader[allReaders.size()];
            for (int i = 0; i < readers.length; i++) {
                readers[i] = allReaders.get(i);
            }

            // always swap the leading reader to the first position
            // For this part we need to make sure we also swap all the other metadata arrays that are indexed by reader index
            if (tempLeadingReaderIdx != 0) {
                final KnnVectorsReader temp = readers[LEADING_READER_IDX];
                readers[LEADING_READER_IDX] = readers[tempLeadingReaderIdx];
                readers[tempLeadingReaderIdx] = temp;
                // also swap the leading doc map to the first position to match the readers
                final MergeState.DocMap tempDocMap = docMaps[LEADING_READER_IDX];
                docMaps[LEADING_READER_IDX] = docMaps[tempLeadingReaderIdx];
                docMaps[tempLeadingReaderIdx] = tempDocMap;
                // swap base ords
                final int tempBaseOrd = baseOrds[LEADING_READER_IDX];
                baseOrds[LEADING_READER_IDX] = baseOrds[tempLeadingReaderIdx];
                baseOrds[tempLeadingReaderIdx] = tempBaseOrd;
            }

            this.perReaderFloatVectorValues = new FloatVectorValues[readers.length];
            this.dimension = dimension;

            // Build mapping from global ordinal to [readerIndex, readerOrd]
            this.ordMapping = new int[totalDocsCount][2];

            int documentsIterated = 0;

            // Will be used to build the new jVectorLuceneDocMap with the new ordinals to docId mapping.
            // This mapping should not be used to access the vectors at any time during construction, but only after the merge is complete
            // and the new segment is created and used by searchers.
            final int[] ordToDocIds = new int[totalLiveVectorsCount];
            this.newToOldOrds = new int[totalLiveVectorsCount];

            // Simulate the merge process to build the ordinal mapping
            // This is similar to what DocIDMerger would do but tracks ordinals

            // For each reader
            int newGlobalOrd = 0;
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
                        final int newGlobalDocId = docMaps[readerIdx].get(docId);
                        final int oldLocalOrd = it.index();
                        final int oldGlobalOrd = oldLocalOrd + baseOrds[readerIdx];
                        ordToDocIds[newGlobalOrd] = newGlobalDocId;
                        newToOldOrds[newGlobalOrd] = oldGlobalOrd;
                        newGlobalOrd++;
                        ordMapping[oldGlobalOrd][READER_ID] = readerIdx; // Reader index
                        ordMapping[oldGlobalOrd][READER_ORD] = oldLocalOrd; // Ordinal in reader
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

            this.jVectorLuceneDocMap = new JVectorLuceneDocMap(ordToDocIds);
            log.debug("Created RandomAccessMergedFloatVectorValues with {} total vectors from {} readers", size, readers.length);

        }

        /**
         * Merges the float vector values from multiple readers into a unified structure.
         * This process includes handling product quantization (PQ) for vector compression,
         * generating ord-to-doc mappings, and writing the merged index into a new segment file.
         * <p>
         * The method determines if pre-existing product quantization codebooks are available
         * from the leading reader. If available, it refines them using remaining vectors
         * from other readers in the merge. If no pre-existing codebooks are found and
         * the total vector count meets the required minimum threshold, new codebooks
         * and compressed vectors are computed. Otherwise, no PQ compression is applied.
         * <p>
         * Also, it generates a mapping of ordinals to document IDs by iterating through
         * the provided vector data, which is further used to write the field data.
         *
         * @throws IOException if there is an issue during reading or writing vector data.
         */
        public void merge() throws IOException {
            // This section creates the PQVectors to be used for this merge
            // Get PQ compressor for leading reader
            final int totalVectorsCount = size;
            final String fieldName = fieldInfo.name;
            final PQVectors pqVectors;
            // Get the leading reader
            PerFieldKnnVectorsFormat.FieldsReader fieldsReader = (PerFieldKnnVectorsFormat.FieldsReader) readers[LEADING_READER_IDX];
            JVectorReader leadingReader = (JVectorReader) fieldsReader.getFieldReader(fieldName);
            // Check if the leading reader has pre-existing PQ codebooks and if so, refine them with the remaining vectors
            if (leadingReader.getProductQuantizationForField(fieldInfo.name).isEmpty()) {
                // No pre-existing codebooks, check if we have enough vectors to trigger quantization
                log.info(
                    "No Pre-existing PQ codebooks found in this merge for field {} in segment {}, will check if a new codebooks is necessary",
                    fieldName,
                    mergeState.segmentInfo.name
                );
                if (this.size() >= minimumBatchSizeForQuantization) {
                    log.info(
                        "Calculating new codebooks and compressed vectors for field: {}, with totalVectorCount: {}, above minimumBatchSizeForQuantization: {}",
                        fieldName,
                        totalVectorsCount,
                        minimumBatchSizeForQuantization
                    );
                    pqVectors = getPQVectors(newToOldOrds, this, fieldInfo);
                } else {
                    log.info(
                        "Not enough vectors found for field: {}, totalVectorCount: {}, is below minimumBatchSizeForQuantization: {}",
                        fieldName,
                        totalVectorsCount,
                        minimumBatchSizeForQuantization
                    );
                    pqVectors = null;
                }
            } else {
                log.info(
                    "Pre-existing PQ codebooks found in this merge for field {} in segment {}, will refine the codebooks from the leading reader with the remaining vectors",
                    fieldName,
                    mergeState.segmentInfo.name
                );
                final long start = Clock.systemDefaultZone().millis();
                ProductQuantization leadingCompressor = leadingReader.getProductQuantizationForField(fieldName).get();
                // Refine the leadingCompressor with the remaining vectors in the merge, we skip the leading reader since it's already been
                // used to create the leadingCompressor
                // We assume the leading reader is ALWAYS the first one in the readers array
                for (int i = LEADING_READER_IDX + 1; i < readers.length; i++) {
                    final FloatVectorValues values = readers[i].getFloatVectorValues(fieldName);
                    final RandomAccessVectorValues randomAccessVectorValues = new RandomAccessVectorValuesOverVectorValues(values);
                    leadingCompressor.refine(randomAccessVectorValues);
                }
                final long end = Clock.systemDefaultZone().millis();
                final long trainingTime = end - start;
                log.info("Refined PQ codebooks for field {}, in {} millis", fieldName, trainingTime);
                KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(trainingTime);
                // pqVectors = leadingCompressor.encodeAll(this, SIMD_POOL);
                pqVectors = PQVectors.encodeAndBuild(leadingCompressor, newToOldOrds.length, newToOldOrds, this, SIMD_POOL);
            }

            writeField(fieldInfo, this, pqVectors, newToOldOrds, jVectorLuceneDocMap);
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

    /**
     * This method will return the graph index for the field
     * @return OnHeapGraphIndex
     */
    public OnHeapGraphIndex getGraph(
        BuildScoreProvider buildScoreProvider,
        RandomAccessVectorValues randomAccessVectorValues,
        int[] newToOldOrds,
        FieldInfo fieldInfo,
        String segmentName
    ) {
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

        log.info("Building graph from merged float vector");
        // parallel graph construction from the merge documents Ids
        SIMD_POOL.submit(() -> IntStream.range(0, newToOldOrds.length).parallel().forEach(ord -> {
            graphIndexBuilder.addGraphNode(ord, vv.get().getVector(newToOldOrds[ord]));
        })).join();
        graphIndexBuilder.cleanup();
        graphIndex = graphIndexBuilder.getGraph();
        final long end = Clock.systemDefaultZone().millis();

        log.info("Built graph for field {} in segment {} in {} millis", fieldInfo.name, segmentName, end - start);
        return graphIndex;
    }

    static class RandomAccessVectorValuesOverVectorValues implements RandomAccessVectorValues {
        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final FloatVectorValues values;

        public RandomAccessVectorValuesOverVectorValues(FloatVectorValues values) {
            this.values = values;
        }

        @Override
        public int size() {
            return values.size();
        }

        @Override
        public int dimension() {
            return values.dimension();
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            try {
                // Access to float values is not thread safe
                synchronized (this) {
                    final float[] vector = values.vectorValue(nodeId);
                    final float[] copy = new float[vector.length];
                    System.arraycopy(vector, 0, copy, 0, vector.length);
                    return VECTOR_TYPE_SUPPORT.createFloatVector(copy);
                }
            } catch (IOException e) {
                log.error("Error retrieving vector at ordinal {}", nodeId, e);
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

    /**
     * This class represents the mapping from the Lucene document IDs to the jVector ordinals.
     * This mapping is necessary because the jVector ordinals can be different from the Lucene document IDs and when lucene documentIDs change after a merge,
     * we need to update this mapping to reflect the new document IDs.
     * This requires us to know the previous mapping from the previous merge and the new mapping from the current merge.
     *
     * Which means that we also need to persist this mapping to disk to be available across merges.
     */
    public static class JVectorLuceneDocMap {
        private static final int VERSION = 1;
        private final int[] ordinalsToDocIds;
        private final int[] docIdsToOrdinals;

        /**
         * Constructor that reads the mapping from the index input
         * @param in The index input
         * @throws IOException if an I/O error occurs
         */
        public JVectorLuceneDocMap(IndexInput in) throws IOException {
            final int version = in.readInt(); // Read the version
            if (version != VERSION) {
                throw new IOException("Unsupported version: " + version);
            }
            int size = in.readVInt();
            int maxDocId = in.readVInt();

            ordinalsToDocIds = new int[size];
            docIdsToOrdinals = new int[maxDocId];
            for (int ord = 0; ord < size; ord++) {
                final int docId = in.readVInt();
                ordinalsToDocIds[ord] = docId;
                docIdsToOrdinals[docId] = ord;
            }
        }

        /**
         * Constructor that creates a new mapping between ordinals and docIds
         * @param ordinalsToDocIds The mapping from ordinals to docIds
         */
        public JVectorLuceneDocMap(int[] ordinalsToDocIds) {
            this.ordinalsToDocIds = new int[ordinalsToDocIds.length];
            System.arraycopy(ordinalsToDocIds, 0, this.ordinalsToDocIds, 0, ordinalsToDocIds.length);
            final int maxDocId = Arrays.stream(ordinalsToDocIds).max().getAsInt();
            final int maxDocs = maxDocId + 1;
            // We are going to assume that the number of ordinals is roughly the same as the number of documents in the segment, therefore,
            // the mapping will not be sparse.
            if (maxDocs < ordinalsToDocIds.length) {
                throw new IllegalStateException("Max docs " + maxDocs + " is less than the number of ordinals " + ordinalsToDocIds.length);
            }
            if (maxDocId > ordinalsToDocIds.length) {
                log.warn(
                    "Max doc id {} is greater than the number of ordinals {}, this implies a lot of deleted documents. Or that some documents are missing vectors. Wasting a lot of memory",
                    maxDocId,
                    ordinalsToDocIds.length
                );
            }
            this.docIdsToOrdinals = new int[maxDocs];
            Arrays.fill(this.docIdsToOrdinals, -1); // -1 means no mapping to ordinal
            for (int ord = 0; ord < ordinalsToDocIds.length; ord++) {
                this.docIdsToOrdinals[ordinalsToDocIds[ord]] = ord;
            }
        }

        /**
         * Updates the mapping from the Lucene document IDs to the jVector ordinals based on the sort operation. (during flush)
         * @param sortMap The sort map
         */
        public void update(Sorter.DocMap sortMap) {
            final int[] newOrdinalsToDocIds = new int[ordinalsToDocIds.length];
            final int[] newDocIdsToOrdinals = new int[docIdsToOrdinals.length];
            Arrays.fill(newDocIdsToOrdinals, -1);
            for (int oldDocId = 0; oldDocId < docIdsToOrdinals.length; oldDocId++) {
                if (docIdsToOrdinals[oldDocId] == -1) {
                    continue;
                }
                final int newDocId = sortMap.oldToNew(oldDocId);
                final int oldOrd = docIdsToOrdinals[oldDocId];
                newDocIdsToOrdinals[newDocId] = oldOrd;
                newOrdinalsToDocIds[oldOrd] = newDocId;
            }
            System.arraycopy(newOrdinalsToDocIds, 0, ordinalsToDocIds, 0, ordinalsToDocIds.length);
            System.arraycopy(newDocIdsToOrdinals, 0, docIdsToOrdinals, 0, docIdsToOrdinals.length);
        }

        /**
         * Returns the jVector ordinal for the given Lucene document ID
         * @param luceneDocId The Lucene document ID
         * @return The jVector ordinal
         */
        public int getJVectorOrdinal(int luceneDocId) {
            return docIdsToOrdinals[luceneDocId];
        }

        /**
         * Returns the Lucene document ID for the given jVector ordinal
         * @param jVectorOrdinal The jVector ordinal
         * @return The Lucene document ID
         *
         * NOTE: This method is useful when, for example, we want to remap acceptedDocs bitmap from Lucene to jVector ordinal bitmap filter
         */
        public int getLuceneDocId(int jVectorOrdinal) {
            return ordinalsToDocIds[jVectorOrdinal];
        }

        /**
         * Writes the mapping to the index output
         * @param out The index output
         * @throws IOException if an I/O error occurs
         */
        public void toOutput(IndexOutput out) throws IOException {
            out.writeInt(VERSION);
            out.writeVInt(ordinalsToDocIds.length);
            out.writeVInt(docIdsToOrdinals.length);
            for (int ord = 0; ord < ordinalsToDocIds.length; ord++) {
                out.writeVInt(ordinalsToDocIds[ord]);
            }
        }
    }
}
