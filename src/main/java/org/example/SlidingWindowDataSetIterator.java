package org.example;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.ArrayList;
import java.util.NoSuchElementException;

public class SlidingWindowDataSetIterator implements DataSetIterator {

    private INDArray data;
    private int windowSize;
    private int labelIndex;
    private int batchSize;
    private int cursor = 0;

    private List<DataSet> windows;

    public SlidingWindowDataSetIterator(
            INDArray data,
            int windowSize,
            int labelIndex,
            int batchSize
    ) {
        this.data = data;
        this.windowSize = windowSize;
        this.labelIndex = labelIndex;
        this.batchSize = batchSize;

        createWindows();
    }

    private void createWindows() {
        windows = new ArrayList<>();

        int rows = data.rows();

        for(int i = 0; i < rows - windowSize; i++) {

            INDArray featureWindow =
                    data.getRows(i, i + windowSize - 1);

            INDArray label =
                    data.getRow(i + windowSize).getColumn(labelIndex);

            INDArray features3d = featureWindow
                    .transpose()
                    .reshape(1, featureWindow.columns(), windowSize);

            INDArray labels2d = label.reshape(1,1);

            windows.add(new DataSet(features3d, labels2d));
        }
    }

    @Override
    public boolean hasNext() {
        return cursor < windows.size();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public DataSet next(int num) {

        if(!hasNext()) {
            throw new NoSuchElementException();
        }

        int end = Math.min(cursor + num, windows.size());

        List<DataSet> batch = windows.subList(cursor, end);

        cursor = end;

        return DataSet.merge(batch);
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    /*
        @Override
        public List<String> getLabels() {
            return List.of();
        }

        @Override
        public int cursor() {
            return cursor;
        }

        @Override
        public int numExamples() {
            return windows.size();
        }
    */
    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }
}