/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import com.google.common.primitives.Floats;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.SetOnce;
import org.apache.lucene.util.StringHelper;
import org.junit.After;
import org.junit.Test;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.lucene.index.OpenSearchLeafReader;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.seqno.SequenceNumbers;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.jvector.BulkJVectorIndexGenerator;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.JVectorKNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.transport.Netty4ModulePlugin;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.opensearch.index.engine.Engine.HISTORY_UUID_KEY;
import static org.opensearch.index.engine.Engine.MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID;
import static org.opensearch.index.translog.Translog.TRANSLOG_UUID_KEY;
import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.codec.jvector.CodecTestsCommon.calculateGroundTruthVectorsIds;
import static org.opensearch.knn.index.engine.CommonTestUtils.PROPERTIES_FIELD_NAME;

/**
 * Integration tests for bulk importing JVector indices into OpenSearch
 * This test demonstrates how to create a Lucene index with JVector offline
 * and then import it into OpenSearch without re-indexing.
 */
@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.TEST, numDataNodes = 1)
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
public class JVectorBulkImportTests extends OpenSearchIntegTestCase {

    private static final String INDEX_NAME = "jvector_imported_index";
    private static final String FIELD_NAME = "vector_field";
    private static final int VECTOR_DIMENSION = 3;
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Override
    protected boolean addMockHttpTransport() {
        return false;
    }

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        // Add the JVector plugin to the cluster
        return List.of(Netty4ModulePlugin.class, JVectorKNNPlugin.class);
    }

    @After
    public void cleanUp() throws IOException {
        if (indexExists(INDEX_NAME)) {
            client().admin().indices().prepareDelete(INDEX_NAME).get();
        }
    }

    /**
     * Test that creates a Lucene index with JVector using BulkJVectorIndexGenerator
     * and then imports it into OpenSearch without re-indexing.
     */
    @Test
    public void testBulkImportJVectorIndex() throws Exception {
        // Step 1: Generate test vectors
        final int k = 10;
        final int numVectors = 100;
        final VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        float[][] vectors = TestUtils.generateRandomVectors(numVectors, VECTOR_DIMENSION);

        // Step 2: Create a temporary directory for the Lucene index
        Path tempIndexDir = Files.createTempDirectory("jvector-temp-index");
        logger.info("Created temporary directory for Lucene index: {}", tempIndexDir);
        
        try {
            // Step 3: Create the OpenSearch index with appropriate settings first
            createOpenSearchIndex();

            // Step 4: Get the existing segment generation to avoid conflicts
            ShardRouting shardRouting = internalCluster().clusterService().state().routingTable().allShards(INDEX_NAME).get(0);
            IndexShard shard = getIndexShard(shardRouting, INDEX_NAME);
            final SegmentInfos preExistingSegmentInfos = shard.store().readLastCommittedSegmentsInfo();
            long nextGeneration = preExistingSegmentInfos.getGeneration() + 1;

            // Step 5: Create the Lucene index with JVector using the correct generation
            createLuceneIndex(tempIndexDir, vectors, vectorSimilarityFunction, nextGeneration);

            // Step 6: Import the Lucene index into OpenSearch
            importLuceneIndex(tempIndexDir);
            
            // Step 6: Verify the index was imported correctly
            verifyImportedIndex(numVectors);
            
            // Step 7: Verify JVector engine is being used
            verifyJVectorEngineIsUsed();
            
            // Step 8: Test search functionality
            testSearchFunctionality(k, vectors, vectorSimilarityFunction);
            
        } finally {
            // Clean up temporary directory
            Files.walk(tempIndexDir)
                .sorted((a, b) -> -a.compareTo(b))
                .forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException e) {
                        logger.warn("Failed to delete temporary file: {}", path, e);
                    }
                });
        }
    }

    private void createLuceneIndex(Path indexPath, float[][] vectors, VectorSimilarityFunction similarityFunction, long nextGeneration) throws IOException {
        logger.info("Creating Lucene index with {} vectors and generation {}", vectors.length, nextGeneration);

        // Convert vectors to RandomAccessVectorValues
        List<VectorFloat<?>> vectorFloatList = new ArrayList<>(vectors.length);
        for (float[] vector : vectors) {
            vectorFloatList.add(VECTOR_TYPE_SUPPORT.createFloatVector(vector));
        }

        RandomAccessVectorValues randomAccessVectorValues = new ListRandomAccessVectorValues(vectorFloatList, VECTOR_DIMENSION);

        // Create the Lucene segment using BulkJVectorIndexGenerator with the specified generation and field name
        BulkJVectorIndexGenerator.createLuceneSegment(indexPath, VECTOR_DIMENSION, randomAccessVectorValues, similarityFunction, nextGeneration, FIELD_NAME);

        logger.info("Successfully created Lucene index at {}", indexPath);
    }

    private void createOpenSearchIndex() throws Exception {
        logger.info("Creating OpenSearch index: {}", INDEX_NAME);
        
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", VECTOR_DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT)
            .startObject(KNN_METHOD)
            .field(KNNConstants.NAME, DISK_ANN)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .startObject(PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, 16)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        
        String mapping = builder.toString();
        Settings indexSettings = Settings.builder()
            .put("index.knn", true)
            .put("index.number_of_shards", 1)
            .put("index.number_of_replicas", 0)
            .build();
        
        createKnnIndex(INDEX_NAME, indexSettings, mapping);
    }

    private void createKnnIndex(String index, Settings settings, String mapping) throws IOException {
        createIndex(index, settings);
        putMappingRequest(index, mapping);
    }

    private void putMappingRequest(String index, String mapping) throws IOException {
        Request request = new Request("PUT", "/" + index + "/_mapping");
        request.setJsonEntity(mapping);
        Response response = getRestClient().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private void importLuceneIndex(Path luceneIndexPath) throws IOException {
        logger.info("Importing Lucene index into OpenSearch");
        
        // Get the OpenSearch data directory for the index
        //final Directory indexDataDirectory = getOpenSearchIndexDataDirectory();
        // Get the shard routing to find the exact path

        ShardRouting shardRouting = internalCluster().clusterService().state().routingTable().allShards(INDEX_NAME).get(0);
        IndexShard shard = getIndexShard(shardRouting, INDEX_NAME);
        final SegmentInfos preExistingSegmentInfos = shard.store().readLastCommittedSegmentsInfo();
        final String translogUUID = preExistingSegmentInfos.getUserData().get(TRANSLOG_UUID_KEY);
        final Path indexDataPath = shard.store().shardPath().getDataPath().resolve("index");
        logger.info("OpenSearch index data path: {}", indexDataPath);

        // First, close the index
        client().admin().indices().prepareClose(INDEX_NAME).get();
        logger.info("Closed index to prepare for recovery");

        try (final Directory luceneIndexDirectory = FSDirectory.open(luceneIndexPath);
             final Directory indexDataDirectory = FSDirectory.open(indexDataPath);) {
            logger.info("Lucene index directory: {}", luceneIndexDirectory);
            final String[] luceneIndexFiles = luceneIndexDirectory.listAll();
            for (String file : indexDataDirectory.listAll()) {
                logger.info("Deleting existing file from OpenSearch index data directory: {}", file);
                indexDataDirectory.deleteFile(file);
            }
            logger.info("Lucene index files to import: {}", luceneIndexFiles);
            for (String file : luceneIndexFiles) {
                // For segments file we are going to modify it's user data and add the translogUUID before writing it
                if (file.startsWith("segments_")) {
                    logger.info("Importing Lucene segments file: {}, to: {}, appending translogUUID: {}", file, indexDataDirectory, translogUUID);
                    // Read the segmentInfos from the lucene index to import and modify it's user data to include the expected translogUUID
                    final SegmentInfos segmentInfos = Lucene.readSegmentInfos(luceneIndexDirectory);
                    final Map<String, String> existingUserData = segmentInfos.getUserData();
                    final Map<String, String> updatedUserData = new HashMap<>(existingUserData);
                    updatedUserData.put(TRANSLOG_UUID_KEY, preExistingSegmentInfos.getUserData().get(TRANSLOG_UUID_KEY));
                    updatedUserData.put(SequenceNumbers.MAX_SEQ_NO, preExistingSegmentInfos.getUserData().get(SequenceNumbers.MAX_SEQ_NO));
                    updatedUserData.put(SequenceNumbers.LOCAL_CHECKPOINT_KEY, preExistingSegmentInfos.getUserData().get(SequenceNumbers.LOCAL_CHECKPOINT_KEY));
                    updatedUserData.put(HISTORY_UUID_KEY, preExistingSegmentInfos.getUserData().get(HISTORY_UUID_KEY));
                    updatedUserData.put(MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID, preExistingSegmentInfos.getUserData().get(MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID));

                    // Fix: Set the next write generation to be greater than the existing max segment generation
                    // This prevents the "Next segment name counter is not greater than max segment name" error
                    long nextGeneration = Math.max(preExistingSegmentInfos.getGeneration() + 1, segmentInfos.getGeneration() + 1);
                    // Ensure we have a positive generation number
                    nextGeneration = Math.max(nextGeneration, 1);
                    //segmentInfos.setNextWriteGeneration(nextGeneration);
                    logger.info("Set next write generation to: {}", nextGeneration);

                    segmentInfos.setUserData(updatedUserData, false);
                    try (final IndexOutput out = indexDataDirectory.createOutput(file, IOContext.DEFAULT)) {
                        segmentInfos.write(out);
                    }
                } else {
                    logger.info("Importing Lucene index file: {}, to: {}", file, indexDataDirectory);
                    indexDataDirectory.copyFrom(luceneIndexDirectory, file, file, IOContext.DEFAULT);
                }

            }
            // write the segmentInfos file with the correct user data (e.g. translogUUID)

            //SegmentCommitInfo segmentCommitInfo = segmentInfos.get().info(0);

            //segmentInfos.get().commit(indexDataDirectory);
            /*
            try (final IndexOutput out = indexDataDirectory.createOutput(file, IOContext.DEFAULT)) {
                segmentInfos.write(out);
            }*/

        }
        logger.info("Successfully imported Lucene index files into OpenSearch index data directory");

        // After copying files, we need to trigger a recovery to make OpenSearch recognize the new files
        
        // Then open it again to trigger recovery from disk
        client().admin().indices().prepareOpen(INDEX_NAME).get();
        logger.info("Opened index to trigger recovery from disk");
        
        // Wait for yellow status at minimum to ensure the recovery is complete
        client().admin().cluster().prepareHealth(INDEX_NAME)
            .setWaitForYellowStatus()
            .setTimeout(TimeValue.timeValueMinutes(1))
            .get();
        logger.info("Index recovery completed");
        
        // Now refresh to make sure all segments are visible
        client().admin().indices().prepareRefresh(INDEX_NAME).get();
        
        logger.info("Successfully imported Lucene index into OpenSearch");
    }

    private void verifyImportedIndex(int expectedDocCount) throws IOException {
        // Refresh the index to ensure all documents are visible
        refresh(INDEX_NAME);

        var response = client().admin().indices().prepareStats(INDEX_NAME).get();
        logger.info("Index stats: {}", response);
        //assertEquals(expectedDocCount, response.getIndices().get(INDEX_NAME).getTotal().getDocs().getCount());

        // Verify the document count
        final SearchResponse searchResponse = client().prepareSearch(INDEX_NAME).setQuery(QueryBuilders.matchAllQuery()).get();
        assertEquals(expectedDocCount, searchResponse.getHits().getTotalHits().value());
        
        logger.info("Successfully verified imported index has {} documents", expectedDocCount);
    }

    private void verifyJVectorEngineIsUsed() throws Exception {
        // Check the mapping to verify JVector engine is specified
        Map<String, Object> indexMapping = getIndexMappingAsMap(INDEX_NAME);
        Map<String, Object> properties = (Map<String, Object>) indexMapping.get(PROPERTIES_FIELD_NAME);
        Map<String, Object> fieldMapping = (Map<String, Object>) properties.get(FIELD_NAME);
        Map<String, Object> methodMapping = (Map<String, Object>) fieldMapping.get(KNNConstants.KNN_METHOD);

        // Verify the engine is set to JVector
        assertEquals(KNNEngine.JVECTOR.getName(), methodMapping.get(KNN_ENGINE));

        // Get index files and verify JVector format is used
        final ShardRouting shardRouting = internalCluster().clusterService().state().routingTable().allShards(INDEX_NAME).get(0);
        try (Engine.Searcher indexSearcher = getIndexShard(shardRouting, INDEX_NAME).acquireSearcher("verify_jvector_engine_is_used")) {
            indexSearcher.getLeafContexts().forEach(leafContext -> {
                // Check the index files to verify JVector codec is being used
                var vectorReader = ((SegmentReader) (((OpenSearchLeafReader) leafContext.reader()).getDelegate())).getVectorReader();
                assertTrue(vectorReader instanceof PerFieldKnnVectorsFormat.FieldsReader);
                var perFieldReader = ((PerFieldKnnVectorsFormat.FieldsReader) vectorReader).getFieldReader(FIELD_NAME);
                assertTrue("JVector codec should be used", perFieldReader instanceof org.opensearch.knn.index.codec.jvector.JVectorReader);
            });
        }
        
        logger.info("Successfully verified JVector engine is being used");
    }

    /**
     * Test the search functionality of the imported index by calculating recall
     *
     * @param k number of nearest neighbors to search for
     * @param vectors vectors that were indexed
     * @param vectorSimilarityFunction similarity function used to index the vectors
     * @throws IOException IOException
     * @throws ParseException ParseException
     */
    private void testSearchFunctionality(int k, float[][] vectors, VectorSimilarityFunction vectorSimilarityFunction) throws IOException, ParseException {
        // Pick a random vector to search with
        final float[] target = TestUtils.generateRandomVectors(1, vectors[0].length)[0];

        // Execute KNN search
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, target, k), k);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        // Verify we got k results (or all docs if less than k)
        assertEquals(Math.min(k, vectors.length), results.size());

        // Parse and verify results
        // Calculate ground truth with brute force
        logger.info("Calculating ground truth");
        final SpaceType spaceType = switch (vectorSimilarityFunction) {
            case EUCLIDEAN -> SpaceType.L2;
            case DOT_PRODUCT -> SpaceType.INNER_PRODUCT;
            case COSINE -> SpaceType.COSINESIMIL;
            default ->
                    throw new IllegalArgumentException("Unsupported similarity function: " + vectorSimilarityFunction);
        };
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(vectors, new float[][] { target }, spaceType, k);
        assertEquals(1, groundTruth.size());
        Set<String> expectedDocIds = groundTruth.getFirst();

        // calculate recall
        logger.info("Calculating recall");
        float recall = ((float) results.stream().filter(r -> expectedDocIds.contains(r.getDocId())).count()) / ((float) k);
        assertTrue("Expected recall to be at least 0.9 but got " + recall, recall >= 0.9);
    }

    private Response searchKNNIndex(String indexName, KNNQueryBuilder knnQueryBuilder, int size) throws IOException {
        Request request = new Request("POST", "/" + indexName + "/_search");
        
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.field("size", size);
        builder.field("query");
        knnQueryBuilder.toXContent(builder, null);
        builder.endObject();
        
        request.setJsonEntity(builder.toString());
        return getRestClient().performRequest(request);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getIndexMappingAsMap(String index) throws Exception {
        Request request = new Request("GET", "/" + index + "/_mapping");
        Response response = getRestClient().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        
        return (Map<String, Object>) ((Map<String, Object>) responseMap.get(index)).get("mappings");
    }

    /**
     * Parse the response of KNN search into a List of KNNResults
     */
    protected List<KNNResult> parseSearchResponse(String responseBody, String fieldName) throws IOException {
        @SuppressWarnings("unchecked")
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(
                MediaTypeRegistry.getDefaultMediaType().xContent(),
                responseBody
        ).map().get("hits")).get("hits");

        @SuppressWarnings("unchecked")
        List<KNNResult> knnSearchResponses = hits.stream().map(hit -> {
            // We are not going to include the float[] vector in the KNNResult since we don't need it for recall calculation, also it's
            // not returned in the response because it's not stored in _source.
            /*final float[] vector = Floats.toArray(
                    Arrays.stream(
                            ((ArrayList<Float>) ((Map<String, Object>) ((Map<String, Object>) hit).get("_source")).get(fieldName)).toArray()
                    ).map(Object::toString).map(Float::valueOf).collect(Collectors.toList())
            );*/
            //TODO: fix this mapping
            //(String) ((Map<String, Object>) hit).get("_id")
            final String id = (String) ((Map<String, Object>) ((Map<String, Object>) hit).get("_source")).get("id");
            return new KNNResult(
                    id,
                    new float[]{},
                    ((Double) ((Map<String, Object>) hit).get("_score")).floatValue()
            );
        }).collect(Collectors.toList());

        return knnSearchResponses;
    }
}