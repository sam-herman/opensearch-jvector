/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.SegmentReader;
import org.junit.Test;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.common.lucene.index.OpenSearchLeafReader;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.engine.Engine;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.plugin.JVectorKNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.transport.Netty4ModulePlugin;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.index.engine.CommonTestUtils.DOC_ID;
import static org.opensearch.knn.index.engine.CommonTestUtils.PROPERTIES_FIELD_NAME;

/**
 * Internal integration tests for k-NN
 * This allows us to not only test rest cases but also get access to the cluster nodes and files
 * This becomes very useful when attempting to detect conditions in the cluster internal state or files
 */

@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.TEST, numDataNodes = 1)
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
public class InternalKNNEngineTests extends OpenSearchIntegTestCase {

    /** ** Enable the http client *** */
    @Override
    protected boolean addMockHttpTransport() {
        return false;
    }

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        // Add the JVector plugin to the cluster
        return List.of(Netty4ModulePlugin.class, JVectorKNNPlugin.class);
    }

    /**
     * Test to validate that the mapping to use JVector engine actually creates the right per field index format with JVector.
     * This test verifies that when JVector engine is specified in the mapping, the index files created use the JVector format.
     */
    @Test
    public void testJVectorEngineCreatesJVectorFormat() throws Exception {
        // Create an index with JVector engine specified in the mapping
        createKnnIndexMappingWithJVectorEngine(CommonTestUtils.DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

        // Add a document with a vector
        Float[] vector = new Float[] { 1.0f, 2.0f, 3.0f };
        client().prepareIndex(INDEX_NAME).setId(DOC_ID).setSource(FIELD_NAME, vector).get();

        // Refresh the index to ensure the document is searchable
        refresh(INDEX_NAME);
        forceMerge(1);

        // Verify the index mapping has JVector engine specified and the JVector engine is being used
        // This checks both the mapping configuration and verifies search functionality
        verifyJVectorEngineIsUsed();
        logger.info("JVector engine should be confirmed through mapping and index format verification");
    }

    private void createKnnIndexMappingWithJVectorEngine(int dimension, SpaceType spaceType, VectorDataType vectorDataType)
        throws Exception {
        String mapping = CommonTestUtils.createIndexMapping(dimension, spaceType, vectorDataType);
        Settings indexSettings = CommonTestUtils.getDefaultIndexSettings();
        // indexSettings = Settings.builder().put(indexSettings).put(INDEX_USE_COMPOUND_FILE.getKey(), false).build();
        createKnnIndex(INDEX_NAME, indexSettings, mapping);
    }

    /**
     * Create KNN Index
     */
    protected void createKnnIndex(String index, Settings settings, String mapping) throws IOException {
        createIndex(index, settings);
        putMappingRequest(index, mapping);
    }

    /**
     * For a given index, make a mapping request
     */
    protected void putMappingRequest(String index, String mapping) throws IOException {
        // Put KNN mapping
        Request request = new Request("PUT", "/" + index + "/_mapping");

        request.setJsonEntity(mapping);
        Response response = getRestClient().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Helper method to verify that the JVector engine is being used correctly.
     * This method verifies that the JVector engine is being used by checking the mapping
     * and validating the codec and per-field format of the files in the index.
     *
     * @return true if JVector engine is confirmed to be in use, false otherwise
     */
    private void verifyJVectorEngineIsUsed() throws Exception {
        // We'll verify the JVector engine is being used by checking:
        // 1. The mapping has a JVector engine specified
        // 2. The files in the index are readable by the JVector codec

        // Check the mapping to verify JVector engine is specified
        Map<String, Object> indexMapping = getIndexMappingAsMap(INDEX_NAME);
        Map<String, Object> properties = (Map<String, Object>) indexMapping.get(PROPERTIES_FIELD_NAME);
        Map<String, Object> fieldMapping = (Map<String, Object>) properties.get(FIELD_NAME);
        Map<String, Object> methodMapping = (Map<String, Object>) fieldMapping.get(KNNConstants.KNN_METHOD);

        // Verify the engine is set to JVector
        assertEquals(KNNEngine.JVECTOR.getName(), methodMapping.get(KNN_ENGINE));

        boolean jvectorEngineInMapping = KNNEngine.JVECTOR.getName().equals(methodMapping.get(KNN_ENGINE));
        logger.info("JVector engine specified in mapping: {}", jvectorEngineInMapping);

        // Get index files
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
    }

    /**
     * Get index mapping as map
     *
     * @param index name of index to fetch
     * @return index mapping a map
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getIndexMappingAsMap(String index) throws Exception {
        Request request = new Request("GET", "/" + index + "/_mapping");

        Response response = getRestClient().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        return (Map<String, Object>) ((Map<String, Object>) responseMap.get(index)).get("mappings");
    }

    /**
     * Test to validate that the JVector engine correctly applies the alpha and neighbor_overflow parameters.
     * This test verifies that when these parameters are specified in the mapping, they are correctly
     * applied to the index configuration.
     */
    @Test
    public void testJVectorEngineWithConstructionParameters() throws Exception {
        // We will use these custom values to verify the parameters are correctly applied
        double alpha = 2.0;
        double neighborOverflow = 1.5;

        // Create an index with JVector engine and custom alpha and neighbor_overflow parameters
        createKnnIndexMappingWithJVectorEngineAndConstructionParams(
            CommonTestUtils.DIMENSION,
            SpaceType.L2,
            VectorDataType.FLOAT,
            alpha, // Custom alpha value
            neighborOverflow  // Custom neighbor_overflow value
        );

        // Add a document with a vector
        Float[] vector = new Float[] { 1.0f, 2.0f, 3.0f };
        client().prepareIndex(INDEX_NAME).setId(DOC_ID).setSource(FIELD_NAME, vector).get();

        // Refresh the index to ensure the document is searchable
        refresh(INDEX_NAME);
        forceMerge(1);

        // Verify the index mapping has the custom construction parameters
        verifyJVectorConstructionParameters(alpha, neighborOverflow);
        logger.info("JVector engine construction parameters verified");
    }

    private void createKnnIndexMappingWithJVectorEngineAndConstructionParams(
        int dimension,
        SpaceType spaceType,
        VectorDataType vectorDataType,
        double alpha,
        double neighborOverflow
    ) throws Exception {
        String mapping = createIndexMappingWithConstructionParams(dimension, spaceType, vectorDataType, alpha, neighborOverflow);
        Settings indexSettings = CommonTestUtils.getDefaultIndexSettings();
        createKnnIndex(INDEX_NAME, indexSettings, mapping);
    }

    private String createIndexMappingWithConstructionParams(
        int dimension,
        SpaceType spaceType,
        VectorDataType vectorDataType,
        double alpha,
        double neighborOverflow
    ) throws IOException {
        try (XContentBuilder builder = XContentFactory.jsonBuilder()) {
            return builder.startObject()
                .startObject(PROPERTIES_FIELD_NAME)
                .startObject(FIELD_NAME)
                .field(CommonTestUtils.TYPE_FIELD_NAME, CommonTestUtils.KNN_VECTOR_TYPE)
                .field(CommonTestUtils.DIMENSION_FIELD_NAME, dimension)
                .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
                .startObject(KNN_METHOD)
                .field(NAME, DISK_ANN)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_M, CommonTestUtils.M)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, CommonTestUtils.EF_CONSTRUCTION)
                .field(METHOD_PARAMETER_ALPHA, alpha)
                .field(METHOD_PARAMETER_NEIGHBOR_OVERFLOW, neighborOverflow)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();
        }
    }

    /**
     * Helper method to verify that the JVector engine construction parameters are correctly applied.
     * This method checks the mapping to ensure the alpha and neighbor_overflow parameters are set correctly.
     */
    private void verifyJVectorConstructionParameters(double expectedAlpha, double expectedNeighborOverflow) throws Exception {
        // Check the mapping to verify construction parameters
        Map<String, Object> indexMapping = getIndexMappingAsMap(INDEX_NAME);
        Map<String, Object> properties = (Map<String, Object>) indexMapping.get(PROPERTIES_FIELD_NAME);
        Map<String, Object> fieldMapping = (Map<String, Object>) properties.get(FIELD_NAME);
        Map<String, Object> methodMapping = (Map<String, Object>) fieldMapping.get(KNN_METHOD);
        Map<String, Object> parameters = (Map<String, Object>) methodMapping.get(PARAMETERS);

        // Verify the construction parameters are set correctly
        assertEquals(expectedAlpha, parameters.get(METHOD_PARAMETER_ALPHA));
        assertEquals(expectedNeighborOverflow, parameters.get(METHOD_PARAMETER_NEIGHBOR_OVERFLOW));

        // Also verify we can search the index to ensure it's functional
        Float[] queryVector = new Float[] { 1.0f, 2.0f, 3.0f };
        Request searchRequest = new Request("GET", "/" + INDEX_NAME + "/_search");
        searchRequest.setJsonEntity(
            "{\n"
                + "  \"query\": {\n"
                + "    \"knn\": {\n"
                + "      \""
                + FIELD_NAME
                + "\": {\n"
                + "        \"vector\": "
                + Arrays.toString(queryVector)
                + ",\n"
                + "        \"k\": 1\n"
                + "      }\n"
                + "    }\n"
                + "  }\n"
                + "}"
        );

        Response searchResponse = getRestClient().performRequest(searchRequest);
        assertEquals(RestStatus.OK, RestStatus.fromCode(searchResponse.getStatusLine().getStatusCode()));
    }
}
