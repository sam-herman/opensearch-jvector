/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.recall;

import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.test.OpenSearchIntegTestCase;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;

/**
 * Tests confirm that for the different supported configurations, recall is sound. The recall thresholds are
 * conservatively and empirically determined to prevent flakiness.
 *
 */
@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.SUITE, numDataNodes = 1)
public class RecallTestsIT extends KNNRestTestCase {

    private static final String PROPERTIES_FIELD = "properties";
    private final static String TEST_INDEX_PREFIX_NAME = "test_index";
    private final static String TEST_FIELD_NAME = "test_field";
    // For more extensive real world scenarios increase the vector dimensions
    private final static int TEST_DIMENSION = 16;
    private final static int DOC_COUNT = 10000;
    private final static int BATCH_SIZE = 1000;
    private final static int QUERY_COUNT = 100;
    private final static int TEST_K = 100;
    private final static double PERFECT_RECALL = 1.0;
    private final static int SHARD_COUNT = 1;
    private final static int REPLICA_COUNT = 0;
    private final static int MAX_SEGMENT_COUNT = 1;

    // Standard algorithm parameters
    private final static int HNSW_M = 16;
    private final static int HNSW_EF_CONSTRUCTION = 100;

    // Setup ground truth for all tests once
    private final static float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, TEST_DIMENSION, true);
    private final static float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, TEST_DIMENSION, DOC_COUNT, true);
    private final static Map<SpaceType, List<Set<String>>> GROUND_TRUTH = Map.of(
        SpaceType.L2,
        TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.L2, TEST_K),
        SpaceType.COSINESIMIL,
        TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.COSINESIMIL, TEST_K),
        SpaceType.INNER_PRODUCT,
        TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.INNER_PRODUCT, TEST_K)
    );
    private static final List<SpaceType> SPACE_TYPE_LIST = List.of(SpaceType.L2, SpaceType.COSINESIMIL);

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {DIMENSION},
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"lucene",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION}
     *          }
     *       }
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenLuceneHnswFP32_thenRecallAbove75percent() {
        for (SpaceType spaceType : SPACE_TYPE_LIST) {
            String indexName = createIndexName(KNNEngine.LUCENE, spaceType);
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(TEST_FIELD_NAME)
                .field(TYPE, TYPE_KNN_VECTOR)
                .field(DIMENSION, TEST_DIMENSION)
                .startObject(KNN_METHOD)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                .field(METHOD_PARAMETER_M, HNSW_M)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
            assertRecall(indexName, spaceType, "pre-merge", 0.25f);
            forceMergeKnnIndex(indexName, MAX_SEGMENT_COUNT);
            assertRecall(indexName, spaceType, "post-merge", 0.25f);
        }
    }

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {TEST_DIMENSION},
     *      "method": {
     *          "name":"diskann",
     *          "engine":"jvector",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION},
     *              "ef_search": {HNSW_EF_SEARCH},
     *          }
     *       }
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenJVectorDiskANN_thenRecallAbove75percent() {
        for (SpaceType spaceType : SPACE_TYPE_LIST) {
            String indexName = createIndexName(KNNEngine.JVECTOR, spaceType);
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(TEST_FIELD_NAME)
                .field(TYPE, TYPE_KNN_VECTOR)
                .field(DIMENSION, TEST_DIMENSION)
                .startObject(KNN_METHOD)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
                .field(NAME, DISK_ANN)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                .field(METHOD_PARAMETER_M, HNSW_M)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
            assertRecall(indexName, spaceType, "pre-merge", 0.25f);
            forceMergeKnnIndex(indexName, MAX_SEGMENT_COUNT);
            assertRecall(indexName, spaceType, "post-merge", 0.25f);
        }
    }

    @SneakyThrows
    private void assertRecall(String testIndexName, SpaceType spaceType, String testType, float acceptableRecallFromPerfect) {
        List<List<String>> searchResults = bulkSearch(testIndexName, TEST_FIELD_NAME, QUERY_VECTORS, TEST_K);
        double recallValue = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH.get(spaceType), TEST_K);
        logger.info("Recall value ({}/{}) = {}", spaceType, testType, recallValue);
        assertEquals(PERFECT_RECALL, recallValue, acceptableRecallFromPerfect);
    }

    private String createIndexName(KNNEngine knnEngine, SpaceType spaceType) {
        return String.format("%s_%s_%s", TEST_INDEX_PREFIX_NAME, knnEngine.getName(), spaceType.getValue());
    }

    @SneakyThrows
    private void createIndexAndIngestDocs(String indexName, String fieldName, Settings settings, String mapping) {
        createKnnIndex(indexName, settings, mapping);
        int i = 0;
        for (; i < DOC_COUNT; i += BATCH_SIZE) {
            logger.info("Ingesting batch {}/{}", i, DOC_COUNT);
            final float[][] indexVectors = new float[BATCH_SIZE][TEST_DIMENSION];
            System.arraycopy(INDEX_VECTORS, i, indexVectors, 0, BATCH_SIZE);
            bulkAddKnnDocs(indexName, fieldName, indexVectors, i, BATCH_SIZE, false);
        }

        if (i < DOC_COUNT) {
            logger.info("Ingesting final batch {}/{}", i, DOC_COUNT);
            final float[][] indexVectors = new float[DOC_COUNT - i][TEST_DIMENSION];
            System.arraycopy(INDEX_VECTORS, i, indexVectors, 0, DOC_COUNT - i);
            bulkAddKnnDocs(indexName, fieldName, indexVectors, i, DOC_COUNT - i, false);
        }

        refreshIndex(indexName);
    }

    private Settings getSettings() {
        return Settings.builder()
            .put("number_of_shards", SHARD_COUNT)
            .put("number_of_replicas", REPLICA_COUNT)
            .put("index.knn", true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
    }
}
