/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.junit.After;
import org.junit.Test;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.common.lucene.index.OpenSearchLeafReader;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.shard.IndexShard;
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

import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
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
        int numVectors = 100;
        List<float[]> vectors = generateTestVectors(numVectors, VECTOR_DIMENSION);
        
        // Step 2: Create a temporary directory for the Lucene index
        Path tempIndexDir = Files.createTempDirectory("jvector-temp-index");
        logger.info("Created temporary directory for Lucene index: {}", tempIndexDir);
        
        try {
            // Step 3: Create the Lucene index with JVector
            createLuceneIndex(tempIndexDir, vectors, VectorSimilarityFunction.EUCLIDEAN);
            
            // Step 4: Create the OpenSearch index with appropriate settings
            createOpenSearchIndex();
            
            // Step 5: Import the Lucene index into OpenSearch
            importLuceneIndex(tempIndexDir);
            
            // Step 6: Verify the index was imported correctly
            verifyImportedIndex(numVectors);
            
            // Step 7: Verify JVector engine is being used
            verifyJVectorEngineIsUsed();
            
            // Step 8: Test search functionality
            testSearchFunctionality(vectors);
            
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

    private List<float[]> generateTestVectors(int count, int dimension) {
        List<float[]> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            float[] vector = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = random().nextFloat();
            }
            vectors.add(vector);
        }
        return vectors;
    }

    private void createLuceneIndex(Path indexPath, List<float[]> vectors, VectorSimilarityFunction similarityFunction) throws IOException {
        logger.info("Creating Lucene index with {} vectors", vectors.size());
        
        // Convert vectors to RandomAccessVectorValues
        List<VectorFloat<?>> vectorFloatList = new ArrayList<>(vectors.size());
        for (float[] vector : vectors) {
            vectorFloatList.add(VECTOR_TYPE_SUPPORT.createFloatVector(vector));
        }
        
        RandomAccessVectorValues randomAccessVectorValues = new ListRandomAccessVectorValues(vectorFloatList, VECTOR_DIMENSION);
        
        // Create the Lucene segment using BulkJVectorIndexGenerator
        BulkJVectorIndexGenerator.createLuceneSegment(indexPath, VECTOR_DIMENSION, randomAccessVectorValues, similarityFunction);
        
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
        final Directory indexDataDirectory = shard.store().directory();
        logger.info("OpenSearch index data directory: {}", indexDataDirectory);

        try (final Directory luceneIndexDirectory = FSDirectory.open(luceneIndexPath)) {
            logger.info("Lucene index directory: {}", luceneIndexDirectory);
            final String[] luceneIndexFiles = luceneIndexDirectory.listAll();
            final Set<String> preExistingIndexDataFiles = Arrays.stream(indexDataDirectory.listAll()).collect(Collectors.toSet());
            logger.info("Lucene index files to import: {}", luceneIndexFiles);
            for (String file : luceneIndexFiles) {
                logger.info("Importing Lucene index file: {}, to: {}", file, indexDataDirectory);
                if (preExistingIndexDataFiles.contains(file)) {
                    logger.info("File already exists in OpenSearch index data directory, deleting to be replaced: {}", file);
                    indexDataDirectory.deleteFile(file);
                }
                indexDataDirectory.copyFrom(luceneIndexDirectory, file, file, IOContext.DEFAULT);
            }
        }
        logger.info("Successfully imported Lucene index files into OpenSearch index data directory");

        // After copying files, we need to trigger a recovery to make OpenSearch recognize the new files
        
        // First, close the index
        client().admin().indices().prepareClose(INDEX_NAME).get();
        logger.info("Closed index to prepare for recovery");
        
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

    private Directory getOpenSearchIndexDataDirectory() {
        // Get the node data path from the environment
        Path nodePath = internalCluster().getInstance(org.opensearch.env.Environment.class).dataFiles()[0];
        logger.info("OpenSearch node data path: {}", nodePath);
        
        // Get the shard routing to find the exact path
        ShardRouting shardRouting = internalCluster().clusterService().state().routingTable().allShards(INDEX_NAME).get(0);
        IndexShard shard = getIndexShard(shardRouting, INDEX_NAME);
        final Path indexDataPath = shard.store().shardPath().getDataPath();
        logger.info("OpenSearch index data path: {}", indexDataPath);

        return shard.store().directory();
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

    private void testSearchFunctionality(List<float[]> vectors) throws IOException, ParseException {
        // Pick a random vector to search with
        float[] queryVector = vectors.get(random().nextInt(vectors.size()));
        int k = 5;
        
        // Execute KNN search
        Response searchResponse = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, k), k);
        assertEquals(RestStatus.OK, RestStatus.fromCode(searchResponse.getStatusLine().getStatusCode()));
        
        // Parse and verify results
        String responseBody = EntityUtils.toString(searchResponse.getEntity());
        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        Map<String, Object> hits = (Map<String, Object>) responseMap.get("hits");
        List<Map<String, Object>> hitsList = (List<Map<String, Object>>) hits.get("hits");
        
        // Verify we got results
        assertFalse("Search should return results", hitsList.isEmpty());
        assertTrue("Search should return at most k results", hitsList.size() <= k);
        
        logger.info("Successfully tested search functionality with {} results", hitsList.size());
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
}