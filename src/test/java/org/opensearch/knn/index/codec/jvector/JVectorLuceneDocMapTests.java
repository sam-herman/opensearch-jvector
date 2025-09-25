/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.index.Sorter;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

import java.io.IOException;

public class JVectorLuceneDocMapTests extends LuceneTestCase {

    @Test
    public void testConstructorWithOrdinalsToDocIds() {
        int[] ordinalsToDocIds = { 0, 2, 4, 6 };
        JVectorLuceneDocMap docMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        // Test ordinal to doc ID mapping
        assertEquals(0, docMap.getLuceneDocId(0));
        assertEquals(2, docMap.getLuceneDocId(1));
        assertEquals(4, docMap.getLuceneDocId(2));
        assertEquals(6, docMap.getLuceneDocId(3));

        // Test doc ID to ordinal mapping
        assertEquals(0, docMap.getJVectorOrdinal(0));
        assertEquals(1, docMap.getJVectorOrdinal(2));
        assertEquals(2, docMap.getJVectorOrdinal(4));
        assertEquals(3, docMap.getJVectorOrdinal(6));

        // Test unmapped doc IDs return -1
        assertEquals(-1, docMap.getJVectorOrdinal(1));
        assertEquals(-1, docMap.getJVectorOrdinal(3));
        assertEquals(-1, docMap.getJVectorOrdinal(5));
    }

    @Test
    public void testConstructorWithSequentialDocIds() {
        int[] ordinalsToDocIds = { 0, 1, 2, 3 };
        JVectorLuceneDocMap docMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        for (int i = 0; i < 4; i++) {
            assertEquals(i, docMap.getLuceneDocId(i));
            assertEquals(i, docMap.getJVectorOrdinal(i));
        }
    }

    @Test(expected = IllegalStateException.class)
    public void testConstructorThrowsWhenMaxDocsLessThanOrdinals() {
        int[] invalidMapping = { -1 }; // This would cause issues in Arrays.stream().max()
        new JVectorLuceneDocMap(invalidMapping);
    }

    @Test
    public void testSerializationRoundTrip() throws IOException {
        int[] ordinalsToDocIds = { 1, 3, 5, 7, 9 };
        JVectorLuceneDocMap originalMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        try (Directory dir = newDirectory()) {
            try (IndexOutput out = dir.createOutput("test", newIOContext(random()))) {
                // Serialize
                originalMap.toOutput(out);
            }

            try (IndexInput in = dir.openInput("test", newIOContext(random()))) {
                // Deserialize
                JVectorLuceneDocMap deserializedMap = new JVectorLuceneDocMap(in);

                // Verify mappings are preserved
                for (int ord = 0; ord < ordinalsToDocIds.length; ord++) {
                    assertEquals(originalMap.getLuceneDocId(ord), deserializedMap.getLuceneDocId(ord));
                }

                for (int docId : ordinalsToDocIds) {
                    assertEquals(originalMap.getJVectorOrdinal(docId), deserializedMap.getJVectorOrdinal(docId));
                }
            }
        }
    }

    @Test(expected = IOException.class)
    public void testDeserializationWithUnsupportedVersion() throws IOException {
        try (Directory dir = newDirectory()) {
            try (IndexOutput out = dir.createOutput("test", newIOContext(random()))) {
                out.writeInt(999); // Unsupported version
                out.writeVInt(1);
                out.writeVInt(1);
                out.writeVInt(0);
            }

            try (IndexInput in = dir.openInput("test", newIOContext(random()))) {
                new JVectorLuceneDocMap(in);
            }
        }
    }

    @Test
    public void testUpdateWithSortMap() {
        int[] ordinalsToDocIds = { 0, 1, 2, 3 };
        JVectorLuceneDocMap docMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        // Create a sort map that reverses the order
        Sorter.DocMap sortMap = new Sorter.DocMap() {
            @Override
            public int oldToNew(int oldDocId) {
                return 3 - oldDocId; // Reverse mapping: 0->3, 1->2, 2->1, 3->0
            }

            @Override
            public int newToOld(int docID) {
                throw new UnsupportedOperationException();
            }

            @Override
            public int size() {
                throw new UnsupportedOperationException();
            }
        };

        docMap.update(sortMap);

        // After update, ordinal 0 should map to doc 3, ordinal 1 to doc 2, etc.
        assertEquals(3, docMap.getLuceneDocId(0));
        assertEquals(2, docMap.getLuceneDocId(1));
        assertEquals(1, docMap.getLuceneDocId(2));
        assertEquals(0, docMap.getLuceneDocId(3));

        // And reverse mappings
        assertEquals(3, docMap.getJVectorOrdinal(0));
        assertEquals(2, docMap.getJVectorOrdinal(1));
        assertEquals(1, docMap.getJVectorOrdinal(2));
        assertEquals(0, docMap.getJVectorOrdinal(3));
    }

    @Test
    public void testUpdateWithSparseMapping() {
        int[] ordinalsToDocIds = { 1, 3, 5 }; // Sparse mapping
        JVectorLuceneDocMap docMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        // Sort map that shifts everything by 1
        Sorter.DocMap sortMap = new Sorter.DocMap() {
            @Override
            public int oldToNew(int oldDocId) {
                return oldDocId + 1;
            }

            @Override
            public int newToOld(int docID) {
                throw new UnsupportedOperationException();
            }

            @Override
            public int size() {
                throw new UnsupportedOperationException();
            }
        };

        docMap.update(sortMap);

        // After update: ordinal 0->doc 2, ordinal 1->doc 4, ordinal 2->doc 6
        assertEquals(2, docMap.getLuceneDocId(0));
        assertEquals(4, docMap.getLuceneDocId(1));
        assertEquals(6, docMap.getLuceneDocId(2));

        // Verify reverse mappings
        assertEquals(0, docMap.getJVectorOrdinal(2));
        assertEquals(1, docMap.getJVectorOrdinal(4));
        assertEquals(2, docMap.getJVectorOrdinal(6));

        // Unmapped docs should still return -1
        assertEquals(-1, docMap.getJVectorOrdinal(0));
        assertEquals(-1, docMap.getJVectorOrdinal(1));
        assertEquals(-1, docMap.getJVectorOrdinal(3));
        assertEquals(-1, docMap.getJVectorOrdinal(5));
    }

    @Test
    public void testEmptyMapping() {
        int[] ordinalsToDocIds = {};
        JVectorLuceneDocMap docMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        // Should handle empty mapping gracefully
    }

    @Test
    public void testLargeDocIdGap() {
        // Test the warning case where maxDocId >> ordinalsToDocIds.length
        int[] ordinalsToDocIds = { 0, 1000000 }; // Large gap
        JVectorLuceneDocMap docMap = new JVectorLuceneDocMap(ordinalsToDocIds);

        assertEquals(0, docMap.getLuceneDocId(0));
        assertEquals(1000000, docMap.getLuceneDocId(1));
        assertEquals(0, docMap.getJVectorOrdinal(0));
        assertEquals(1, docMap.getJVectorOrdinal(1000000));

        // All docs in between should return -1
        assertEquals(-1, docMap.getJVectorOrdinal(500000));
    }
}
