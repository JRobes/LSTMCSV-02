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
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App 
{
    private static int lines2Skip = 1;
    private static int minibatchSize = 4; //Number of examples in each minibatch
    private static int windowSize = 7;
    private static int labelColumn = 0;
    private static double percentageOfTest = 0.2; //Entero de 1 a 100
    public static void main( String[] args ) throws IOException, InterruptedException {
        System.out.println( "Hello World!" );
        List<String> lines = Files.readAllLines(Paths.get("test-2.csv"));
        System.out.println("En archivo csv número total de líneas: " + lines.size());
        int totalSamples = lines.size()-lines2Skip;
        int numFeatures = lines.get(lines2Skip).split(",").length;
        System.out.println("Num lineas con datos: " + totalSamples + "\tNúmero features: " + numFeatures);
        //double[][] matrix = new double[totalSamples][numFeatures];
        String[][] matrix = new String[totalSamples][numFeatures];

        for (int i = lines2Skip; i < totalSamples; i++) { //saltar las lineas indicadas

            String[] values = lines.get(i).split(",");
            for (int j = 0; j < numFeatures; j++) {

                matrix[i - lines2Skip][j] = (values[j]);

                //matrix[i - lines2Skip][j] = Double.parseDouble(values[j]);
                //System.out.print(matrix[i][j] + "\t");
            }
            //System.out.println();
        }
        if(windowSize + 1 > totalSamples){
            System.out.println("El número total de muestras + 1 es menor que el tamaño de la ventana, no sale ni una secuencia... se cierra el programa");
            return;
        }
        int numSequences = totalSamples - windowSize + 1 - 1; //Ojo hay que quitar la primera linea que solo label
        System.out.println("Número de secuencias: " + numSequences);
        int testSequences = (int)(numSequences*percentageOfTest);
        System.out.println("Secuencias de test: " + testSequences);
        int trainSequences = numSequences - testSequences;

        //INDArray testLabels = Nd4j.create(minibatchSize, 1, windowSize);
        String[][][] testLabels = new String[minibatchSize][1][windowSize];
        for (int i = 0; i < minibatchSize; i++) {
            for (int t = 0; t < windowSize; t++) {
                testLabels[i][0][t] = matrix[i+t][0];
            }
        }
        //System.out.println("Test labels shape: " + Arrays.toString(testLabels));
        //System.out.println(testLabels.);
        for (int i = 0; i < testLabels.length; i++) {
            for (int j = 0; j < testLabels[i].length; j++) {
                for (int k = 0; k < testLabels[i][j].length; k++) {
                    System.out.println("[" + i + "][" + j + "][" + k + "] = " + testLabels[i][j][k]);
                }
            }
        }


        /*
        INDArray trainLabels = Nd4j.create(numSequences - testSequences, 1, windowSize);
        for (int i = 0; i < trainSequences; i++) {
            for (int t = 0; t < windowSize; t++) {

                trainLabels.putScalar(new int[]{i, 0, t}, matrix[i + testSequences +t][0]);
            }
        }
        //System.out.println("Train labels shape: " + Arrays.toString(trainLabels.shape()));
        //System.out.println(trainLabels);

        INDArray testData = Nd4j.create(testSequences,numFeatures, windowSize);
        for (int i = 0; i < testSequences; i++) {
            for (int t = 0; t < windowSize; t++) {
                for (int f = 0; f < 4; f++) {
                    testData.putScalar(new int[]{i, f, t}, matrix[i+t][f]);
                 }
            }
        }
        //System.out.println("Test data shape: " + Arrays.toString(testData.shape()));
        //System.out.println(testData);
*/



        //INDArray data = Nd4j.create(matrix);
        //System.out.println(data);

        //DataSetIterator iterator = new SlidingWindowIterator(data, minibatchSize, windowSize, labelColumn);
        //iterator.next();

        //Si tu red es RNN/LSTM, el shape suele ser:
        //[batchSize, numFeatures, timeSteps]
        //Se comprueba con:
        //System.out.println(Arrays.toString(ds.getFeatures().shape()));

        //InputSplit inputSplit = new FileSplit(baseDir);
        //int numLinesToSkip = 0; //Optional, allows us to skip header lines
        //String delimiter = ","; //Comma-delimited files
        //SequenceRecordReader reader = new CSVSequenceRecordReader(numLinesToSkip, delimiter);

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
