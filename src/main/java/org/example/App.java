package org.example;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws IOException, InterruptedException {
        System.out.println( "Hello World!" );
        List<String> lines = Files.readAllLines(Paths.get("Gold Price.csv"));
        System.out.println(lines.size());
        int N = lines.size();
        int M = lines.get(0).split(",").length;
        System.out.println("N: " + N + "\tM: " + M);
        double[][] matrix = new double[N][M];
        for (int i = 1; i < N; i++) { //saltar una linea
            String[] values = lines.get(i).split(",");
            for (int j = 0; j < M; j++) {
                matrix[i][j] = Double.parseDouble(values[j]);
            }
        }
        INDArray data = Nd4j.create(matrix);
        //System.out.println(data);
        int minibatchSize = 10; //Number of examples in each minibatch
        int windowSize = 60;
        int labelColumn = 0;
        DataSetIterator iterator = new SlidingWindowIterator(data, minibatchSize, windowSize, labelColumn);

        //Si tu red es RNN/LSTM, el shape suele ser:
        //[batchSize, numFeatures, timeSteps]
        //Se comprueba con:
        //System.out.println(Arrays.toString(ds.getFeatures().shape()));



        // We are using a random number generator to randomize the order

        //InputSplit inputSplit = new FileSplit(baseDir);
        int numLinesToSkip = 0; //Optional, allows us to skip header lines
        String delimiter = ","; //Comma-delimited files
        SequenceRecordReader reader = new CSVSequenceRecordReader(numLinesToSkip, delimiter);

/*

        DataSetIterator iterator =
                new SequenceRecordReaderDataSetIterator(reader, minibatchSize, labelIndex, numClasses);

        DataSetIterator iterator2 =
                new SequenceRecordReaderDataSetIterator(reader,);

        SequenceRecordReaderDataSetIterator iterator =
                new SequenceRecordReaderDataSetIterator(
                        trainFeatures,
                        trainLabels,
                        minibatchSize,
                        -1,      // no clasificación
                        true     // regresión
                );
*/
        //myNetwork.fit(iterator);
    }
}
