package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.junit.Assert;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

public class CodecTestsCommon {
    public static final String TEST_FIELD = "test_field";
    public static final String TEST_ID_FIELD = "id";

    public static float calculateRecall(IndexReader reader, Set<Integer> groundTruthVectorsIds, TopDocs topDocs, int k)
            throws IOException {
        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        Assert.assertEquals(groundTruthVectorsIds.size(), scoreDocs.length);
        int totalRelevantDocs = 0;
        for (ScoreDoc scoreDoc : scoreDocs) {
            final int id = reader.storedFields().document(scoreDoc.doc).getField(TEST_ID_FIELD).storedValue().getIntValue();
            if (groundTruthVectorsIds.contains(id)) {
                totalRelevantDocs++;
            }
        }
        return ((float) totalRelevantDocs) / ((float) k);
    }

    /**
     * Find the IDs of the ground truth vectors in the dataset
     * @param query query vector
     * @param dataset dataset of all the vectors with their ordinal position in the array as their ID
     * @param k the number of expected results
     * @return the IDs of the ground truth vectors in the dataset
     */
    public static Set<Integer> calculateGroundTruthVectorsIds(
            float[] query,
            final float[][] dataset,
            int k,
            VectorSimilarityFunction vectorSimilarityFunction
    ) {
        final Set<Integer> groundTruthVectorsIds = new HashSet<>();
        final PriorityQueue<ScoreDoc> priorityQueue = new PriorityQueue<>(k, (o1, o2) -> Float.compare(o1.score, o2.score));
        for (int i = 0; i < dataset.length; i++) {
            ScoreDoc scoreDoc = new ScoreDoc(i, vectorSimilarityFunction.compare(query, dataset[i]));
            if (priorityQueue.size() >= k) {
                final ScoreDoc top = priorityQueue.poll();
                if (top.score < scoreDoc.score) {
                    priorityQueue.add(scoreDoc);
                } else {
                    priorityQueue.add(top);
                }
            } else {
                priorityQueue.add(scoreDoc);
            }
        }
        while (!priorityQueue.isEmpty()) {
            groundTruthVectorsIds.add(priorityQueue.poll().doc);
        }

        return groundTruthVectorsIds;
    }

    public static JVectorKnnFloatVectorQuery getJVectorKnnFloatVectorQuery(String fieldName, float[] target, int k, Query filterQuery) {
        return getJVectorKnnFloatVectorQuery(fieldName, target, k, filterQuery, KNNConstants.DEFAULT_OVER_QUERY_FACTOR);
    }

    public static JVectorKnnFloatVectorQuery getJVectorKnnFloatVectorQuery(
            String fieldName,
            float[] target,
            int k,
            Query filterQuery,
            int overQueryFactor
    ) {
        return new JVectorKnnFloatVectorQuery(
                fieldName,
                target,
                k,
                filterQuery,
                overQueryFactor,
                KNNConstants.DEFAULT_QUERY_SIMILARITY_THRESHOLD.floatValue(),
                KNNConstants.DEFAULT_QUERY_RERANK_FLOOR.floatValue(),
                KNNConstants.DEFAULT_QUERY_USE_PRUNING
        );
    }
}
