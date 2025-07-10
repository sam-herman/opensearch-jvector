package org.opensearch.knn.index.codec.jvector;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskSequentialGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Clock;
import java.util.*;
import java.util.stream.IntStream;

import static org.opensearch.knn.index.codec.jvector.CodecTestsCommon.TEST_FIELD;
import static org.opensearch.knn.index.codec.jvector.CodecTestsCommon.calculateGroundTruthVectorsIds;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL;

@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@Log4j2
public class JVectorLuceneConvertorTests extends LuceneTestCase {
    private static final String SEGMENT_NAME = "_bulk";
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();


    /**
     * Creates a complete Lucene segment with all necessary metadata files
     */
    @Test
    public void createLuceneSegment() throws IOException {
        final int numDocs = 3;
        final int dimension = 16;
        final org.apache.lucene.index.VectorSimilarityFunction vectorSimilarityFunction = org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

        final float[][] vectors = new float[numDocs][dimension];
        for (int i = 0; i < numDocs; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random().nextFloat();
            }
        }
        final List<VectorFloat<?>> vectorFloatList = new ArrayList<>(vectors.length);
        for (int i = 0; i < vectors.length; i++) {
            vectorFloatList.add(VECTOR_TYPE_SUPPORT.createFloatVector(vectors[i]));
        }

        final Path indexDirectoryPath = createTempDir("converted-lucene-index");
        final RandomAccessVectorValues randomAccessVectorValues = new ListRandomAccessVectorValues(vectorFloatList, dimension);
        BulkJVectorIndexGenerator.createLuceneSegment(indexDirectoryPath, dimension, randomAccessVectorValues, vectorSimilarityFunction);

        testCreatedIndex(indexDirectoryPath, numDocs, vectors, 2, vectorSimilarityFunction);
    }

    // Verify the index is created properly and check recall
    private static void testCreatedIndex(final Path indexDirectoryPath, int expectedNumDocs, final float[][] vectors, int k, org.apache.lucene.index.VectorSimilarityFunction vectorSimilarityFunction) throws IOException {
        log.info("Attempting to re-open the Lucene index created earlier");

        final float[] target = TestUtils.generateRandomVectors(1, vectors[0].length)[0];
        final Set<Integer> groundTruthVectorsIds = calculateGroundTruthVectorsIds(target, vectors, k, vectorSimilarityFunction);

        try (Directory directory = FSDirectory.open(indexDirectoryPath);
             IndexReader reader = DirectoryReader.open(directory)) {
            log.info("Successfully opened the created Lucene index with {} documents", reader.numDocs());

            log.info("We should now have a single segment with {} documents", expectedNumDocs);
            Assert.assertEquals(1, reader.getContext().leaves().size());
            Assert.assertEquals(expectedNumDocs, reader.numDocs());

            final Query filterQuery = new MatchAllDocsQuery();
            final IndexSearcher searcher = newSearcher(reader);
            KnnFloatVectorQuery knnFloatVectorQuery = CodecTestsCommon.getJVectorKnnFloatVectorQuery(CodecTestsCommon.TEST_FIELD, target, k, filterQuery);
            TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
            assertEquals(k, topDocs.totalHits.value());
            final float recall = CodecTestsCommon.calculateRecall(reader, groundTruthVectorsIds, CodecTestsCommon.TEST_FIELD, topDocs, k);
            Assert.assertEquals(1.0f, recall, 0.05f);
            log.info("successfully completed search tests");
        }
    }
}
