/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Arrays;

/**
 * This class represents the mapping from the Lucene document IDs to the jVector ordinals.
 * This mapping is necessary because the jVector ordinals can be different from the Lucene document IDs and when lucene documentIDs change after a merge,
 * we need to update this mapping to reflect the new document IDs.
 * This requires us to know the previous mapping from the previous merge and the new mapping from the current merge.
 * <p>
 * Which means that we also need to persist this mapping to disk to be available across merges.
 */
@Log4j2
public class JVectorLuceneDocMap {
    private static final int VERSION = 1;
    private int[] ordinalsToDocIds;
    private int[] docIdsToOrdinals;

    /**
     * Constructor that reads the mapping from the index input
     *
     * @param in The index input
     * @throws IOException if an I/O error occurs
     */
    public JVectorLuceneDocMap(IndexInput in) throws IOException {
        final int version = in.readInt(); // Read the version
        if (version != VERSION) {
            throw new IOException("Unsupported version: " + version);
        }
        int size = in.readVInt();
        int maxDocId = in.readVInt();

        ordinalsToDocIds = new int[size];
        docIdsToOrdinals = new int[maxDocId];
        for (int ord = 0; ord < size; ord++) {
            final int docId = in.readVInt();
            ordinalsToDocIds[ord] = docId;
            docIdsToOrdinals[docId] = ord;
        }
    }

    /**
     * Constructor that creates a new mapping between ordinals and docIds
     *
     * @param ordinalsToDocIds The mapping from ordinals to docIds
     */
    public JVectorLuceneDocMap(int[] ordinalsToDocIds) {
        if (ordinalsToDocIds.length == 0) {
            this.ordinalsToDocIds = new int[0];
            this.docIdsToOrdinals = new int[0];
            return;
        }
        this.ordinalsToDocIds = new int[ordinalsToDocIds.length];
        System.arraycopy(ordinalsToDocIds, 0, this.ordinalsToDocIds, 0, ordinalsToDocIds.length);
        final int maxDocId = Arrays.stream(ordinalsToDocIds).max().getAsInt();
        final int maxDocs = maxDocId + 1;
        // We are going to assume that the number of ordinals is roughly the same as the number of documents in the segment, therefore,
        // the mapping will not be sparse.
        if (maxDocs < ordinalsToDocIds.length) {
            throw new IllegalStateException("Max docs " + maxDocs + " is less than the number of ordinals " + ordinalsToDocIds.length);
        }
        if (maxDocId > ordinalsToDocIds.length) {
            log.warn(
                "Max doc id {} is greater than the number of ordinals {}, this implies a lot of deleted documents. Or that some documents are missing vectors. Wasting a lot of memory",
                maxDocId,
                ordinalsToDocIds.length
            );
        }
        this.docIdsToOrdinals = new int[maxDocs];
        Arrays.fill(this.docIdsToOrdinals, -1); // -1 means no mapping to ordinal
        for (int ord = 0; ord < ordinalsToDocIds.length; ord++) {
            this.docIdsToOrdinals[ordinalsToDocIds[ord]] = ord;
        }
    }

    /**
     * Updates the mapping from the Lucene document IDs to the jVector ordinals based on the sort operation. (during flush)
     *
     * @param sortMap The sort map
     */
    public void update(Sorter.DocMap sortMap) {
        final int[] newOrdinalsToDocIds = new int[ordinalsToDocIds.length];
        final int maxNewDocId = Arrays.stream(ordinalsToDocIds).map(sortMap::oldToNew).max().getAsInt();
        final int maxDocs = maxNewDocId + 1;
        if (maxDocs < ordinalsToDocIds.length) {
            throw new IllegalStateException("Max docs " + maxDocs + " is less than the number of ordinals " + ordinalsToDocIds.length);
        }
        final int[] newDocIdsToOrdinals = new int[maxDocs];
        Arrays.fill(newDocIdsToOrdinals, -1);
        for (int oldDocId = 0; oldDocId < docIdsToOrdinals.length; oldDocId++) {
            if (docIdsToOrdinals[oldDocId] == -1) {
                continue;
            }
            final int newDocId = sortMap.oldToNew(oldDocId);
            final int oldOrd = docIdsToOrdinals[oldDocId];
            newDocIdsToOrdinals[newDocId] = oldOrd;
            newOrdinalsToDocIds[oldOrd] = newDocId;
        }
        this.docIdsToOrdinals = newDocIdsToOrdinals;
        this.ordinalsToDocIds = newOrdinalsToDocIds;
    }

    /**
     * Returns the jVector ordinal for the given Lucene document ID
     *
     * @param luceneDocId The Lucene document ID
     * @return The jVector ordinal
     */
    public int getJVectorOrdinal(int luceneDocId) {
        return docIdsToOrdinals[luceneDocId];
    }

    /**
     * Returns the Lucene document ID for the given jVector ordinal
     *
     * @param jVectorOrdinal The jVector ordinal
     * @return The Lucene document ID
     * <p>
     * NOTE: This method is useful when, for example, we want to remap acceptedDocs bitmap from Lucene to jVector ordinal bitmap filter
     */
    public int getLuceneDocId(int jVectorOrdinal) {
        return ordinalsToDocIds[jVectorOrdinal];
    }

    /**
     * Writes the mapping to the index output
     *
     * @param out The index output
     * @throws IOException if an I/O error occurs
     */
    public void toOutput(IndexOutput out) throws IOException {
        out.writeInt(VERSION);
        out.writeVInt(ordinalsToDocIds.length);
        out.writeVInt(docIdsToOrdinals.length);
        for (int ord = 0; ord < ordinalsToDocIds.length; ord++) {
            out.writeVInt(ordinalsToDocIds[ord]);
        }
    }
}
