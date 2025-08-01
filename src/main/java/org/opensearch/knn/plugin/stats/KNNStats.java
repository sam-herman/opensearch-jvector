/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.plugin.stats.suppliers.KNNCounterSupplier;
import org.opensearch.knn.plugin.stats.suppliers.LibraryInitializedSupplier;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Class represents all stats the plugin keeps track of
 */
public class KNNStats {

    private final Map<String, KNNStat<?>> knnStats;

    /**
     * Constructor
     */
    public KNNStats() {
        this.knnStats = buildStatsMap();
    }

    /**
     * Get the stats
     *
     * @return all of the stats
     */
    public Map<String, KNNStat<?>> getStats() {
        return knnStats;
    }

    /**
     * Get a map of the stats that are kept at the node level
     *
     * @return Map of stats kept at the node level
     */
    public Map<String, KNNStat<?>> getNodeStats() {
        return getClusterOrNodeStats(false);
    }

    /**
     * Get a map of the stats that are kept at the cluster level
     *
     * @return Map of stats kept at the cluster level
     */
    public Map<String, KNNStat<?>> getClusterStats() {
        return getClusterOrNodeStats(true);
    }

    private Map<String, KNNStat<?>> getClusterOrNodeStats(Boolean getClusterStats) {
        Map<String, KNNStat<?>> statsMap = new HashMap<>();

        for (Map.Entry<String, KNNStat<?>> entry : knnStats.entrySet()) {
            if (entry.getValue().isClusterLevel() == getClusterStats) {
                statsMap.put(entry.getKey(), entry.getValue());
            }
        }
        return statsMap;
    }

    private Map<String, KNNStat<?>> buildStatsMap() {
        ImmutableMap.Builder<String, KNNStat<?>> builder = ImmutableMap.<String, KNNStat<?>>builder();
        addQueryStats(builder);
        addEngineStats(builder);
        addScriptStats(builder);
        addGraphStats(builder);
        return builder.build();
    }

    private void addQueryStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        // KNN Query Stats
        builder.put(StatNames.KNN_QUERY_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_REQUESTS)))
            .put(
                StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS))
            );

        // Min Score Query Stats
        builder.put(
            StatNames.MIN_SCORE_QUERY_REQUESTS.getName(),
            new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MIN_SCORE_QUERY_REQUESTS))
        )
            .put(
                StatNames.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS))
            );

        // Max Distance Query Stats
        builder.put(
            StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName(),
            new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MAX_DISTANCE_QUERY_REQUESTS))
        )
            .put(
                StatNames.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS))
            );

        // K-NN search, indexing and merge stats
        builder.put(
            StatNames.KNN_QUERY_VISITED_NODES.getName(),
            new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_VISITED_NODES))
        )
            .put(
                StatNames.KNN_QUERY_RERANKED_COUNT.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_RERANKED_COUNT))
            )
            .put(
                StatNames.KNN_QUERY_EXPANDED_NODES.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_EXPANDED_NODES))
            )
            .put(
                StatNames.KNN_QUERY_EXPANDED_BASE_LAYER_NODES.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_EXPANDED_BASE_LAYER_NODES))
            )
            .put(
                StatNames.KNN_QUERY_GRAPH_SEARCH_TIME.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_GRAPH_SEARCH_TIME))
            )
            .put(
                StatNames.KNN_QUANTIZATION_TRAINING_TIME.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUANTIZATION_TRAINING_TIME))
            )
            .put(StatNames.KNN_GRAPH_MERGE_TIME.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_GRAPH_MERGE_TIME)));
    }

    private void addEngineStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.LUCENE_LOADED.getName(), new KNNStat<>(false, new LibraryInitializedSupplier(KNNEngine.LUCENE)));
    }

    private void addScriptStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.SCRIPT_COMPILATIONS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATIONS)))
            .put(
                StatNames.SCRIPT_COMPILATION_ERRORS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATION_ERRORS))
            )
            .put(StatNames.SCRIPT_QUERY_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_REQUESTS)))
            .put(StatNames.SCRIPT_QUERY_ERRORS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_ERRORS)));
    }

    private void addGraphStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.GRAPH_STATS.getName(), new KNNStat<>(false, new Supplier<Map<String, Map<String, Object>>>() {
            @Override
            public Map<String, Map<String, Object>> get() {
                return createGraphStatsMap();
            }
        }));
    }

    private Map<String, Map<String, Object>> createGraphStatsMap() {
        Map<String, Object> mergeMap = new HashMap<>();
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_OPERATIONS.getName(), KNNGraphValue.MERGE_CURRENT_OPERATIONS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_DOCS.getName(), KNNGraphValue.MERGE_CURRENT_DOCS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.getName(), KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_OPERATIONS.getName(), KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getName(), KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_DOCS.getName(), KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getName(), KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
        Map<String, Object> refreshMap = new HashMap<>();
        refreshMap.put(KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getName(), KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        refreshMap.put(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getName(), KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue());
        Map<String, Map<String, Object>> graphStatsMap = new HashMap<>();
        graphStatsMap.put(StatNames.MERGE.getName(), mergeMap);
        graphStatsMap.put(StatNames.REFRESH.getName(), refreshMap);
        return graphStatsMap;
    }
}
