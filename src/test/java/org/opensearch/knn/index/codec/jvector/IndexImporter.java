/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.action.admin.cluster.health.ClusterHealthRequest;
import org.opensearch.action.admin.cluster.health.ClusterHealthResponse;
import org.opensearch.action.admin.indices.open.OpenIndexRequest;
import org.opensearch.action.admin.indices.refresh.RefreshRequest;
import org.opensearch.client.Request;
import org.opensearch.client.RequestOptions;
import org.opensearch.client.Response;
import org.opensearch.client.RestHighLevelClient;
import org.opensearch.client.indices.CloseIndexRequest;
import org.opensearch.client.indices.GetIndexRequest;
import org.opensearch.client.indices.GetIndexResponse;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.cluster.health.ClusterHealthStatus;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContent;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.seqno.SequenceNumbers;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.opensearch.index.engine.Engine.HISTORY_UUID_KEY;
import static org.opensearch.index.engine.Engine.MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID;
import static org.opensearch.index.translog.Translog.TRANSLOG_UUID_KEY;

/**
 * A helper class that can import an existing Lucene index into an existing shard.
 * This can become quite useful for the purposes of bulk loading indices after we generate a large vector graph offline for example
 * using {@link BulkJVectorIndexGenerator}
 */
@Log4j2
public class IndexImporter {
    private static final NamedXContentRegistry DEFAULT_NAMED_X_CONTENT_REGISTRY = new NamedXContentRegistry(
        ClusterModule.getNamedXWriteables()
    );

    private final RestHighLevelClient client;
    private final String indexName;
    private final String dataPath;

    public IndexImporter(RestHighLevelClient client, String indexName) throws IOException, ParseException {
        this.client = client;
        this.indexName = indexName;
        this.dataPath = getBaseDataPathsForIndex(indexName).getFirst();
        log.info("Created IndexImporter with indexName: {}, dataPath: {}", indexName, dataPath);
    }

    public void importLuceneIndex(Path luceneIndexPath) throws IOException {
        log.info("Importing Lucene index into OpenSearch");
        final Path indexDataPath = getOpenSearchIndexDataDirectory();
        log.info("OpenSearch index data directory: {}", indexDataPath);
        // First, close the index
        CloseIndexRequest closeRequest = new CloseIndexRequest(indexName);
        client.indices().close(closeRequest, RequestOptions.DEFAULT);

        log.info("Closed index to prepare for recovery");

        try (
            final Directory luceneIndexDirectory = FSDirectory.open(luceneIndexPath);
            final Directory indexDataDirectory = FSDirectory.open(indexDataPath);
        ) {
            log.info("Lucene index directory: {}", luceneIndexDirectory);
            final SegmentInfos preExistingSegmentInfos = Lucene.readSegmentInfos(indexDataDirectory);
            final String[] luceneIndexFiles = luceneIndexDirectory.listAll();
            for (String file : indexDataDirectory.listAll()) {
                if (!file.contains("write.lock")) {
                    log.info("Deleting existing file from OpenSearch index data directory: {}", file);
                    indexDataDirectory.deleteFile(file);
                }

            }
            log.info("Lucene index files to import: {}", luceneIndexFiles);
            for (String file : luceneIndexFiles) {
                // For segments file we are going to modify it's user data and add the translogUUID before writing it
                if (file.startsWith("segments_")) {
                    log.info("Importing Lucene segments file: {}, to: {}", file, indexDataDirectory);
                    // Read the segmentInfos from the lucene index to import and modify it's user data to include the expected translogUUID
                    final SegmentInfos segmentInfos = Lucene.readSegmentInfos(luceneIndexDirectory);
                    final Map<String, String> existingUserData = segmentInfos.getUserData();
                    final Map<String, String> updatedUserData = new HashMap<>(existingUserData);
                    updatedUserData.put(TRANSLOG_UUID_KEY, preExistingSegmentInfos.getUserData().get(TRANSLOG_UUID_KEY));
                    updatedUserData.put(SequenceNumbers.MAX_SEQ_NO, preExistingSegmentInfos.getUserData().get(SequenceNumbers.MAX_SEQ_NO));
                    updatedUserData.put(
                        SequenceNumbers.LOCAL_CHECKPOINT_KEY,
                        preExistingSegmentInfos.getUserData().get(SequenceNumbers.LOCAL_CHECKPOINT_KEY)
                    );
                    updatedUserData.put(HISTORY_UUID_KEY, preExistingSegmentInfos.getUserData().get(HISTORY_UUID_KEY));
                    updatedUserData.put(
                        MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID,
                        preExistingSegmentInfos.getUserData().get(MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID)
                    );

                    // Fix: Set the next write generation to be greater than the existing max segment generation
                    // This prevents the "Next segment name counter is not greater than max segment name" error
                    long nextGeneration = Math.max(preExistingSegmentInfos.getGeneration() + 1, segmentInfos.getGeneration() + 1);
                    // Ensure we have a positive generation number
                    nextGeneration = Math.max(nextGeneration, 1);
                    // segmentInfos.setNextWriteGeneration(segmentInfos.getLastGeneration() + 1);
                    log.info("Set next write generation to: {}", nextGeneration);

                    segmentInfos.setUserData(updatedUserData, false);
                    try (final IndexOutput out = indexDataDirectory.createOutput(file, IOContext.DEFAULT)) {
                        segmentInfos.write(out);
                    }
                } else {
                    log.info("Importing Lucene index file: {}, to: {}", file, indexDataDirectory);
                    indexDataDirectory.copyFrom(luceneIndexDirectory, file, file, IOContext.DEFAULT);
                }

            }

        }
        log.info("Successfully imported Lucene index files into OpenSearch index data directory");

        // After copying files, we need to trigger a recovery to make OpenSearch recognize the new files

        // Then open it again to trigger recovery from disk
        OpenIndexRequest openRequest = new OpenIndexRequest(indexName);
        client.indices().open(openRequest, RequestOptions.DEFAULT);

        log.info("Opened index to trigger recovery from disk");

        // Wait for yellow status at minimum to ensure the recovery is complete
        ClusterHealthRequest healthRequest = new ClusterHealthRequest(indexName);
        healthRequest.waitForStatus(ClusterHealthStatus.YELLOW);
        healthRequest.timeout(TimeValue.timeValueMinutes(1));
        ClusterHealthResponse healthResponse = client.cluster().health(healthRequest, RequestOptions.DEFAULT);

        log.info("Index recovery completed");

        // Now refresh to make sure all segments are visible
        RefreshRequest refreshRequest = new RefreshRequest(indexName);
        client.indices().refresh(refreshRequest, RequestOptions.DEFAULT);

        log.info("Successfully imported Lucene index into OpenSearch");
    }

    private Path getOpenSearchIndexDataDirectory() throws IOException {
        // Construct the path: {data_path}/nodes/0/indices/{index_uuid}/0/index
        Path basePath = Paths.get(dataPath, "indices");

        // You'll need to get the index UUID from cluster state or settings
        String indexUuid = getIndexUuid();

        return basePath.resolve(indexUuid).resolve("0").resolve("index");

    }

    private String getIndexUuid() throws IOException {
        // Use OpenSearch client to get index metadata
        GetIndexRequest request = new GetIndexRequest(indexName);
        GetIndexResponse response = client.indices().get(request, RequestOptions.DEFAULT);

        return response.getSettings().get(indexName).get("index.uuid");
    }

    /**
     * Takes indexName as an argument and returns the base data path for it's primary shards
     * @param indexName the index name
     * @return base data path for index primary shards
     * @throws IOException
     */
    private List<String> getBaseDataPathsForIndex(String indexName) throws IOException, ParseException {

        Map<String, List<ShardInfo>> shardInfoMap = getIndexShardsFromStats(indexName);
        List<String> dataPathsForIndex = shardInfoMap.values()
            .stream()
            .flatMap(Collection::stream)
            .map(shardInfo -> shardInfo.dataPath)
            .toList();
        if (dataPathsForIndex.isEmpty()) {
            throw new IOException("No data paths found for nodes containing shards of index: " + indexName);
        }

        return dataPathsForIndex;
    }

    private Map<String, List<ShardInfo>> getIndexShardsFromStats(String indexName) throws IOException, ParseException {
        // Get index stats
        Request request = new Request("GET", "/" + indexName + "/_stats?level=shards&pretty");
        Response response = client.getLowLevelClient().performRequest(request);
        String responseBody = EntityUtils.toString(response.getEntity());

        // Parse the JSON response
        @SuppressWarnings("unchecked")
        Map<String, Object> responseMap = (Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map();

        // Navigate to the shards section
        @SuppressWarnings("unchecked")
        Map<String, Object> indices = (Map<String, Object>) responseMap.get("indices");
        @SuppressWarnings("unchecked")
        Map<String, Object> indexData = (Map<String, Object>) indices.get(indexName);
        @SuppressWarnings("unchecked")
        Map<String, Object> shards = (Map<String, Object>) indexData.get("shards");

        Map<String, List<ShardInfo>> shardInfoMap = new HashMap<>();

        // Parse each shard
        for (Map.Entry<String, Object> shardEntry : shards.entrySet()) {
            String shardId = shardEntry.getKey();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> shardInstances = (List<Map<String, Object>>) shardEntry.getValue();

            List<ShardInfo> shardInfoList = new ArrayList<>();
            for (Map<String, Object> shardInstance : shardInstances) {
                ShardInfo shardInfo = parseShardInfo(shardInstance);
                shardInfoList.add(shardInfo);
            }

            shardInfoMap.put(shardId, shardInfoList);
        }

        return shardInfoMap;
    }

    private ShardInfo parseShardInfo(Map<String, Object> shardData) {
        @SuppressWarnings("unchecked")
        Map<String, Object> routing = (Map<String, Object>) shardData.get("routing");

        String node = (String) routing.get("node");
        String state = (String) routing.get("state");
        Boolean primary = (Boolean) routing.get("primary");
        String relocatingNode = (String) routing.get("relocating_node");
        Map<String, Object> shardPath = (Map<String, Object>) shardData.get("shard_path");
        String dataPath = (String) shardPath.get("data_path");
        String statePath = (String) shardPath.get("state_path");

        return new ShardInfo(node, state, primary, relocatingNode, dataPath, statePath);
    }

    // Helper class to hold shard information
    public static class ShardInfo {
        private final String node;
        private final String state;
        private final boolean primary;
        private final String relocatingNode;
        private final String dataPath;
        private final String statePath;

        public ShardInfo(String node, String state, Boolean primary, String relocatingNode, String dataPath, String statePath) {
            this.node = node;
            this.state = state;
            this.primary = primary != null ? primary : false;
            this.relocatingNode = relocatingNode;
            this.dataPath = dataPath;
            this.statePath = statePath;
        }

        public String getNode() {
            return node;
        }

        public String getState() {
            return state;
        }

        public boolean isPrimary() {
            return primary;
        }

        public String getRelocatingNode() {
            return relocatingNode;
        }

        public String getDataPath() {
            return dataPath;
        }

        public String getStatePath() {
            return statePath;
        }

        @Override
        public String toString() {
            return String.format(
                "ShardInfo{node='%s', state='%s', primary=%s, relocatingNode='%s, dataPath='%s', statePath='%s'}",
                node,
                state,
                primary,
                relocatingNode,
                dataPath,
                statePath
            );
        }
    }

    private XContentParser createParser(XContent xContent, String data) throws IOException {
        return xContent.createParser(DEFAULT_NAMED_X_CONTENT_REGISTRY, LoggingDeprecationHandler.INSTANCE, data);
    }
}
