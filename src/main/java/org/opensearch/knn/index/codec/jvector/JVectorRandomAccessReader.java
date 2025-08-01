/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Log4j2
public class JVectorRandomAccessReader implements RandomAccessReader {
    private final byte[] internalBuffer = new byte[Long.BYTES];
    private final byte[] internalFloatBuffer = new byte[Float.BYTES];
    private final IndexInput indexInputDelegate;
    private volatile boolean closed = false;

    public JVectorRandomAccessReader(IndexInput indexInputDelegate) {
        this.indexInputDelegate = indexInputDelegate;
    }

    @Override
    public void seek(long offset) throws IOException {
        indexInputDelegate.seek(offset);
    }

    @Override
    public long getPosition() throws IOException {
        return indexInputDelegate.getFilePointer();
    }

    @Override
    public int readInt() throws IOException {
        return indexInputDelegate.readInt();
    }

    @Override
    public float readFloat() throws IOException {
        indexInputDelegate.readBytes(internalFloatBuffer, 0, Float.BYTES);
        FloatBuffer buffer = ByteBuffer.wrap(internalFloatBuffer).asFloatBuffer();
        return buffer.get();
    }

    // TODO: bring back to override when upgrading jVector again
    // @Override
    public long readLong() throws IOException {
        return indexInputDelegate.readLong();
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        indexInputDelegate.readBytes(bytes, 0, bytes.length);
    }

    @Override
    public void readFully(ByteBuffer buffer) throws IOException {
        // validate that the requested bytes actually exist ----
        long remainingInFile = indexInputDelegate.length() - indexInputDelegate.getFilePointer();
        if (buffer.remaining() > remainingInFile) {
            throw new EOFException("Requested " + buffer.remaining() + " bytes but only " + remainingInFile + " available");
        }

        // Heap buffers with a backing array can be filled in one call ----
        if (buffer.hasArray()) {
            int off = buffer.arrayOffset() + buffer.position();
            int len = buffer.remaining();
            indexInputDelegate.readBytes(buffer.array(), off, len);
            buffer.position(buffer.limit());           // advance fully
            return;
        }

        // Direct / non-array buffers: copy in reasonable chunks ----
        while (buffer.hasRemaining()) {
            final int bytesToRead = Math.min(buffer.remaining(), Long.BYTES);
            indexInputDelegate.readBytes(this.internalBuffer, 0, bytesToRead);
            buffer.put(this.internalBuffer, 0, bytesToRead);
        }
    }

    @Override
    public void readFully(long[] vector) throws IOException {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = readLong();
        }
    }

    @Override
    public void read(int[] ints, int offset, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            ints[offset + i] = readInt();
        }
    }

    @Override
    public void read(float[] floats, int offset, int count) throws IOException {
        final ByteBuffer byteBuffer = ByteBuffer.allocate(Float.BYTES * count);
        indexInputDelegate.readBytes(byteBuffer.array(), offset, Float.BYTES * count);
        FloatBuffer buffer = byteBuffer.asFloatBuffer();
        buffer.get(floats, offset, count);
    }

    @Override
    public void close() throws IOException {
        log.debug("Closing JVectorRandomAccessReader for file: {}", indexInputDelegate);
        this.closed = true;
        // no need to really close the index input delegate since it is a clone
        log.debug("Closed JVectorRandomAccessReader for file: {}", indexInputDelegate);
    }

    @Override
    public long length() throws IOException {
        return indexInputDelegate.length();
    }

    /**
     * Supplies readers which are actually slices of the original IndexInput.
     * We will vend out slices in order for us to easily find the footer of the jVector graph index.
     * This is useful because our logic that reads the graph that the footer is always at {@link IndexInput#length()} of the slice.
     * Which is how {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex#load(ReaderSupplier, long)} is working behind the scenes.
     * The header offset, on the other hand, is flexible because we can provide it as a parameter to {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex#load(ReaderSupplier, long)}
     */
    public static class Supplier implements ReaderSupplier {
        private final AtomicInteger readerCount = new AtomicInteger(0);
        private final IndexInput currentInput;
        private final long sliceStartOffset;
        private final long sliceLength;
        private final ConcurrentHashMap<Integer, RandomAccessReader> readers = new ConcurrentHashMap<>();

        public Supplier(IndexInput indexInput, long sliceStartOffset, long sliceLength) throws IOException {
            this.currentInput = indexInput;
            this.sliceStartOffset = sliceStartOffset;
            this.sliceLength = sliceLength;
        }

        @Override
        public RandomAccessReader get() throws IOException {
            synchronized (this) {
                final IndexInput input = currentInput.slice("Input Slice for the jVector graph or PQ", sliceStartOffset, sliceLength)
                    .clone();

                var reader = new JVectorRandomAccessReader(input);
                int readerId = readerCount.getAndIncrement();
                readers.put(readerId, reader);
                return reader;
            }

        }

        @Override
        public void close() throws IOException {
            // Close source of all cloned inputs
            IOUtils.closeWhileHandlingException(currentInput);

            // Close all readers
            for (RandomAccessReader reader : readers.values()) {
                IOUtils.closeWhileHandlingException(reader::close);
            }
            readers.clear();
            readerCount.set(0);
        }
    }
}
