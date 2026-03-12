package org.example;
import java.util.List;
import java.util.NoSuchElementException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class SlidingWindowIterator implements DataSetIterator {

    private INDArray data;              // [timeSteps, features]
    private int windowSize;
    private int batchSize;
    private int cursor = 0;
    private int numExamples;
    private int numFeatures;

    private DataSetPreProcessor preProcessor;

    public SlidingWindowIterator(INDArray data, int windowSize, int batchSize) {
        this.data = data;
        this.windowSize = windowSize;
        this.batchSize = batchSize;

        this.numFeatures = (int) data.size(1);
        this.numExamples = (int) data.size(0) - windowSize;
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public DataSet next(int num) {

        if (!hasNext()) {
            throw new NoSuchElementException();
        }

        int actualBatch = Math.min(num, numExamples - cursor);

        INDArray features = Nd4j.create(actualBatch, numFeatures, windowSize);
        INDArray labels = Nd4j.create(actualBatch, numFeatures);

        for (int i = 0; i < actualBatch; i++) {

            int start = cursor + i;
            int end = start + windowSize;

            INDArray window = data.getRows(start, end - 1).transpose();

            features.putRow(i, window);

            INDArray label = data.getRow(end);
            labels.putRow(i, label);
        }

        cursor += actualBatch;

        DataSet ds = new DataSet(features, labels);

        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }

        return ds;
    }

    @Override
    public int inputColumns() {
        return numFeatures;
    }

    @Override
    public int totalOutcomes() {
        return numFeatures;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }
/*
    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return numExamples;
    }
*/
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }
}