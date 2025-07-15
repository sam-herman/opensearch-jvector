package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNNCodecVersion;

public class JVectorCodecUtils {
    public static Codec getCodec() {
        return getCodec(JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION, JVectorFormat.DEFAULT_MERGE_ON_DISK);
    }

    /**
     * Helper method to get a codec that uses JVectorFormat as a per-field format
     * @return codec that uses JVectorFormat as a per-field format
     */
    public static Codec getCodec(int minBatchSizeForQuantization, boolean mergeOnDisk) {
        // Create a custom PerFieldKnnVectorsFormat that returns JVectorFormat for our test field
        final PerFieldKnnVectorsFormat perFieldFormat = new PerFieldKnnVectorsFormat() {
            @Override
            public org.apache.lucene.codecs.KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new JVectorFormat(minBatchSizeForQuantization, mergeOnDisk);
            }
        };

        // Create a custom codec that uses the PerFieldKnnVectorsFormat
        org.apache.lucene.codecs.Codec baseCodec = new org.apache.lucene.codecs.lucene101.Lucene101Codec();
        KNNCodecVersion codecVersion = KNNCodecVersion.current();
        return new org.apache.lucene.codecs.FilterCodec(codecVersion.getCodecName(), baseCodec) {
            @Override
            public org.apache.lucene.codecs.KnnVectorsFormat knnVectorsFormat() {
                return perFieldFormat;
            }
        };
    }
}
