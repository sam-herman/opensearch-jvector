/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskSequentialGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.*;
import org.opensearch.index.mapper.Uid;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.common.bytes.BytesReference;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Clock;
import java.util.*;
import java.util.stream.IntStream;

import static org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY;
import static org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat.PER_FIELD_SUFFIX_KEY;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL;

/**
 * Helper tool to bulk load vectors into a jVector graph and later wrap it in a Lucene index.
 * This can be used to generate a large jVector index for testing and benchmarking purposes.
 * The benefits of the approach help us in the following way:
 * <nl>
 *      <li>Avoid merge overhead otherwise associated with indexing via Lucene/OpenSearch </li>
 *      <li>Avoid batching logic</li>
 *      <li>Ability to generate large index offline by maximizing concurrency on a given machine</li>
 * </nl>
 */
@Log4j2
public class BulkJVectorIndexGenerator {
    private static final String DEFAULT_SEGMENT_NAME = "_bulk";
    private static final String SOURCE_FIELD_NAME = "_source";
    private static final String ID_FIELD_NAME = "_id";

    public static void createLuceneSegment(
        Path indexDirectoryPath,
        int dimension,
        RandomAccessVectorValues randomAccessVectorValues,
        org.apache.lucene.index.VectorSimilarityFunction vectorSimilarityFunction
    ) throws IOException {
        createLuceneSegment(indexDirectoryPath, dimension, randomAccessVectorValues, vectorSimilarityFunction, 1);
    }

    public static void createLuceneSegment(
        Path indexDirectoryPath,
        int dimension,
        RandomAccessVectorValues randomAccessVectorValues,
        org.apache.lucene.index.VectorSimilarityFunction vectorSimilarityFunction,
        long nextWriteGeneration
    ) throws IOException {
        createLuceneSegment(
            indexDirectoryPath,
            dimension,
            randomAccessVectorValues,
            vectorSimilarityFunction,
            nextWriteGeneration,
            CodecTestsCommon.TEST_FIELD
        );
    }

    public static void createLuceneSegment(
        Path indexDirectoryPath,
        int dimension,
        RandomAccessVectorValues randomAccessVectorValues,
        org.apache.lucene.index.VectorSimilarityFunction vectorSimilarityFunction,
        long nextWriteGeneration,
        String fieldName
    ) throws IOException {
        // create the Lucene index with the jVector vectors field
        try (Directory directory = FSDirectory.open(indexDirectoryPath)) {

            // Create FieldInfo and FieldInfos
            Map<String, String> vectorFieldAttributes = new HashMap<>();
            vectorFieldAttributes.put(PER_FIELD_FORMAT_KEY, JVectorFormat.NAME);
            vectorFieldAttributes.put(PER_FIELD_SUFFIX_KEY, Long.toString(0));
            final FieldInfo vectorFieldInfo = new FieldInfo(
                fieldName,
                0, // field number
                false, // storeTermVector
                false, // omitNorms
                false, // storePayloads
                IndexOptions.NONE, // indexOptions
                DocValuesType.NONE, // docValuesType
                DocValuesSkipIndexType.NONE, // docValues skip index
                -1, // dvGen
                vectorFieldAttributes, // attributes
                0, // pointDimensionCount
                0, // pointIndexDimensionCount
                0, // pointNumBytes
                dimension, // vectorDimension
                VectorEncoding.FLOAT32, // vectorEncoding
                vectorSimilarityFunction, // vectorSimilarityFunction
                false, // softDeletes
                false  // parentField
            );

            final FieldInfo sourceFieldInfo = new FieldInfo(
                SOURCE_FIELD_NAME,
                1, // field number
                false, // storeTermVector
                false, // omitNorms
                true, // storePayloads
                IndexOptions.NONE, // indexOptions
                DocValuesType.NONE, // docValuesType
                DocValuesSkipIndexType.NONE, // docValues skip index
                -1, // dvGen
                new HashMap<>(), // attributes
                0, // pointDimensionCount
                0, // pointIndexDimensionCount
                0, // pointNumBytes
                0, // vectorDimension
                VectorEncoding.FLOAT32, // vectorEncoding
                vectorSimilarityFunction, // vectorSimilarityFunction
                false, // softDeletes
                false  // parentField
            );

            final FieldInfo idFieldInfo = new FieldInfo(
                ID_FIELD_NAME,
                2, // field number
                false, // storeTermVector
                false, // omitNorms
                true, // storePayloads
                IndexOptions.NONE, // indexOptions
                DocValuesType.NONE, // docValuesType
                DocValuesSkipIndexType.NONE, // docValues skip index
                -1, // dvGen
                new HashMap<>(), // attributes
                0, // pointDimensionCount
                0, // pointIndexDimensionCount
                0, // pointNumBytes
                0, // vectorDimension
                VectorEncoding.FLOAT32, // vectorEncoding
                vectorSimilarityFunction, // vectorSimilarityFunction
                false, // softDeletes
                false  // parentField
            );

            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { vectorFieldInfo, sourceFieldInfo, idFieldInfo });

            // Add required stored fields format mode to segment attributes
            Map<String, String> segmentAttributes = new HashMap<>();
            segmentAttributes.put("Lucene90StoredFieldsFormat.mode", "BEST_SPEED");
            SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                DEFAULT_SEGMENT_NAME,
                randomAccessVectorValues.size(), // maxDoc
                false, // useCompoundFile
                false, // has blocks
                JVectorCodecUtils.getCodec(), // codec
                Collections.emptyMap(), // diagnostics
                UUID.randomUUID().toString().substring(0, 16).getBytes(), // id
                segmentAttributes, // attributes
                null // indexSort
            );

            segmentInfo.setFiles(new HashSet<>());

            // Create SegmentWriteState
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.getDefault(),
                directory,
                segmentInfo,
                fieldInfos,
                null, // pendingDeletes
                IOContext.DEFAULT,
                ""
            );

            final OnHeapGraphIndex preBuiltGraph = buildJVectorIndex(
                randomAccessVectorValues,
                JVectorWriter.getVectorSimilarityFunction(vectorFieldInfo)
            );

            // Create a SegmentWriteState for the vector fields
            final String vectorFieldsSegmentSuffix = JVectorFormat.NAME + "_" + Long.toString(0);
            final SegmentWriteState vectorFieldsWriteState = new SegmentWriteState(writeState, vectorFieldsSegmentSuffix);

            // Write flat vectors (required by JVectorWriter)
            writeFlatVectors(vectorFieldsWriteState, vectorFieldInfo, randomAccessVectorValues);

            // Write jVector index using pre-built graph
            writeJVectorIndex(vectorFieldsWriteState, vectorFieldInfo, preBuiltGraph, randomAccessVectorValues);

            // Write stored fields for _source field and _id field
            writeStoredFields(writeState, randomAccessVectorValues.size(), sourceFieldInfo, idFieldInfo);

            // Write field infos
            writeFieldInfos(directory, fieldInfos, segmentInfo);

            // After all files are written (before creating SegmentInfos), collect all segment files:
            Set<String> segmentFiles = new HashSet<>();
            for (String file : directory.listAll()) {
                if (file.startsWith(segmentInfo.name)) {
                    segmentFiles.add(file);
                }
            }

            // Update segment info with all files
            segmentInfo.setFiles(segmentFiles);
            // Write segment info
            writeSegmentInfo(directory, segmentInfo);

            // Create and write SegmentInfos (this creates the segments_1 file)
            SegmentInfos segmentInfos = new SegmentInfos(Version.LATEST.major);
            SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, -1, -1, -1, StringHelper.randomId());
            segmentInfos.add(segmentCommitInfo);
            segmentInfos.setNextWriteGeneration(nextWriteGeneration);
            segmentInfos.commit(directory);

            log.info("Created complete Lucene segment with jVector index");
        }
    }

    /**
     * Writes the segment info file
     */
    private static void writeSegmentInfo(Directory directory, SegmentInfo segmentInfo) throws IOException {
        segmentInfo.getCodec().segmentInfoFormat().write(directory, segmentInfo, IOContext.DEFAULT);
    }

    /**
     * Writes the field infos file
     */
    private static void writeFieldInfos(Directory directory, FieldInfos fieldInfos, SegmentInfo segmentInfo) throws IOException {
        segmentInfo.getCodec().fieldInfosFormat().write(directory, segmentInfo, "", fieldInfos, IOContext.DEFAULT);
    }

    /**
     * Writes stored fields with proper _source field containing document data and _id field containing document id
     */
    private static void writeStoredFields(SegmentWriteState writeState, int numDocs, FieldInfo sourceFieldInfo, FieldInfo idFieldInfo)
        throws IOException {
        try (
            StoredFieldsWriter storedFieldsWriter = writeState.segmentInfo.getCodec()
                .storedFieldsFormat()
                .fieldsWriter(writeState.directory, writeState.segmentInfo, writeState.context)
        ) {

            for (int i = 0; i < numDocs; i++) {
                final String id = Integer.toString(i, 10);

                storedFieldsWriter.startDocument();

                // Create source document with id field
                Map<String, Object> sourceDoc = new HashMap<>();
                sourceDoc.put("id", id);

                // Convert to JSON bytes using OpenSearch's XContentBuilder
                XContentBuilder builder = MediaTypeRegistry.contentBuilder(MediaTypeRegistry.JSON).map(sourceDoc);
                BytesReference sourceBytes = BytesReference.bytes(builder);

                // Write the _source field as binary
                storedFieldsWriter.writeField(sourceFieldInfo, sourceBytes.toBytesRef());

                // Write the _id field as string
                final BytesRef encodedId = Uid.encodeId(id);
                storedFieldsWriter.writeField(idFieldInfo, encodedId);
                storedFieldsWriter.finishDocument();
            }

            storedFieldsWriter.finish(numDocs);
        }
    }

    /**
     * Writes flat vectors using Lucene's flat vector format
     */
    private static void writeFlatVectors(
        SegmentWriteState writeState,
        FieldInfo fieldInfo,
        RandomAccessVectorValues randomAccessVectorValues
    ) throws IOException {
        FlatVectorsFormat flatFormat = new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
        FlatVectorsWriter flatWriter = flatFormat.fieldsWriter(writeState);
        FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) flatWriter.addField(fieldInfo);

        // Add all vectors to the flat writer
        for (int i = 0; i < randomAccessVectorValues.size(); i++) {
            final float[] vector = new float[randomAccessVectorValues.dimension()];
            for (int j = 0; j < randomAccessVectorValues.dimension(); j++) {
                vector[j] = randomAccessVectorValues.getVector(i).get(j);
            }
            fieldWriter.addValue(i, vector);
        }
        flatWriter.flush(randomAccessVectorValues.size() - 1, null);
        flatWriter.finish();
        IOUtils.close(flatWriter);
    }

    /**
     * Writes the jVector index using the pre-built graph
     */
    private static void writeJVectorIndex(
        SegmentWriteState writeState,
        FieldInfo fieldInfo,
        OnHeapGraphIndex preBuiltGraph,
        RandomAccessVectorValues randomAccessVectorValues
    ) throws IOException {

        final JVectorWriter.FieldWriter<float[]> fieldWriter = new JVectorWriter.FieldWriter<>(
            fieldInfo,
            DEFAULT_SEGMENT_NAME,
            null,
            randomAccessVectorValues,
            JVectorFormat.DEFAULT_MAX_CONN,
            JVectorFormat.DEFAULT_BEAM_WIDTH,
            KNNConstants.DEFAULT_NEIGHBOR_OVERFLOW_VALUE.floatValue(),
            KNNConstants.DEFAULT_ALPHA_VALUE.floatValue()
        );

        final var vectorIndexFieldMetadata = writeGraph(preBuiltGraph, fieldWriter, null, writeState);
        String metaFileName = IndexFileNames.segmentFileName(
            writeState.segmentInfo.name,
            writeState.segmentSuffix,
            JVectorFormat.META_EXTENSION
        );
        try (IndexOutput meta = writeState.directory.createOutput(metaFileName, writeState.context)) {
            CodecUtil.writeIndexHeader(
                meta,
                JVectorFormat.META_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                writeState.segmentInfo.getId(),
                writeState.segmentSuffix
            );

            meta.writeInt(fieldWriter.getFieldInfo().number);
            vectorIndexFieldMetadata.toOutput(meta);

            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
    }

    /**
     * Builds a complete jVector OnHeapGraphIndex using the jVector library
     */
    private static OnHeapGraphIndex buildJVectorIndex(
        RandomAccessVectorValues vectorValues,
        VectorSimilarityFunction vectorSimilarityFunction
    ) throws IOException {
        log.info("Building jVector graph index with {} vectors", vectorValues.size());

        final long start = Clock.systemDefaultZone().millis();

        BuildScoreProvider buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(vectorValues, vectorSimilarityFunction);

        GraphIndexBuilder graphIndexBuilder = new GraphIndexBuilder(
            buildScoreProvider,
            vectorValues.dimension(),
            JVectorFormat.DEFAULT_MAX_CONN,
            JVectorFormat.DEFAULT_BEAM_WIDTH,
            KNNConstants.DEFAULT_NEIGHBOR_OVERFLOW_VALUE.floatValue(),
            KNNConstants.DEFAULT_ALPHA_VALUE.floatValue(),
            true
        );

        // Build graph in parallel
        var vv = vectorValues.threadLocalSupplier();
        int size = vectorValues.size();

        SIMD_POOL.submit(() -> {
            IntStream.range(0, size).parallel().forEach(node -> { graphIndexBuilder.addGraphNode(node, vv.get().getVector(node)); });
        }).join();

        graphIndexBuilder.cleanup();
        OnHeapGraphIndex graph = graphIndexBuilder.getGraph();

        final long end = Clock.systemDefaultZone().millis();
        log.info("Built jVector graph index in {} ms", end - start);

        return graph;
    }

    private static JVectorWriter.VectorIndexFieldMetadata writeGraph(
        OnHeapGraphIndex graph,
        JVectorWriter.FieldWriter<?> fieldData,
        PQVectors pqVectors,
        SegmentWriteState segmentWriteState
    ) throws IOException {
        // field data file, which contains the graph
        final String baseDataFileName = segmentWriteState.segmentInfo.name + "_" + segmentWriteState.segmentSuffix;
        final String vectorIndexFieldFileName = baseDataFileName
            + "_"
            + fieldData.getFieldInfo().name
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
            var resultBuilder = JVectorWriter.VectorIndexFieldMetadata.builder()
                .fieldNumber(fieldData.getFieldInfo().number)
                .vectorEncoding(fieldData.getFieldInfo().getVectorEncoding())
                .vectorSimilarityFunction(fieldData.getFieldInfo().getVectorSimilarityFunction())
                .vectorDimension(fieldData.getRandomAccessVectorValues().dimension());

            try (
                var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, jVectorIndexWriter).with(
                    new InlineVectors(fieldData.getRandomAccessVectorValues().dimension())
                ).build()
            ) {
                var suppliers = Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(fieldData.getRandomAccessVectorValues().getVector(nodeId))
                );
                writer.write(suppliers);
                long endGraphOffset = jVectorIndexWriter.position();
                resultBuilder.vectorIndexOffset(startOffset);
                resultBuilder.vectorIndexLength(endGraphOffset - startOffset);

                // If PQ is enabled and we have enough vectors, write the PQ codebooks and compressed vectors
                if (pqVectors != null) {
                    log.info(
                        "Writing PQ codebooks and vectors for field {} since the size is {} >= {}",
                        fieldData.getFieldInfo().name,
                        fieldData.getRandomAccessVectorValues().size(),
                        JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION
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
}
