package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    private static double percentageOfTrain = 0.8; //Entero de 1 a 100
    private static int windowSize = 7;
    private static int minibatchSize = 4; //Number of examples in each minibatch
    private static int labelColumn = 0;
    public static void main( String[] args ) throws IOException, InterruptedException {
        System.out.println( "Hello World!" );
        List<String> lines = Files.readAllLines(Paths.get("test.csv"));
        System.out.println("En archivo csv número total de líneas: " + lines.size());
        System.out.println("Lines to Skip: " + lines2Skip);
        System.out.println("Percent of train: " + percentageOfTrain);
        System.out.println("window Size: " + windowSize);
        System.out.println("Minibatch size: " + minibatchSize);
        for(int i = 0; i < lines2Skip; i++){
            lines.remove(0);
        }
        int totalSamples   = lines.size();
        int trainSequences = (int)(totalSamples * percentageOfTrain);
        int testSequences  = totalSamples - trainSequences;
        System.out.println("Total Samples:\t\t"  + totalSamples);
        System.out.println("Train Samples:\t\t"  + trainSequences);
        System.out.println("Test Sequences:\t\t" + testSequences);
        int numFeatures = lines.get(0).split(",").length;
        System.out.println("Num features:" + numFeatures);
        List<String> testList = lines.subList(totalSamples - testSequences - windowSize + 1 ,lines.size());
        System.out.println("Lineas en lista de test: " + testList.size());
        List<String> trainList = lines.subList(0 ,trainSequences);
        System.out.println("Lineas en lista de train: " + trainList.size());

        INDArray dataTest = getIndArray(testList);
        long[] shape = dataTest.shape();
        System.out.println("INDArray test: " + Arrays.toString(shape));
        INDArray dataTrain = getIndArray(trainList);
        long[] shape2 = dataTrain.shape();
        System.out.println("INDArray train: " + Arrays.toString(shape2));


    }

    private static INDArray getIndArray(List<String> lines) {
        int numRows = lines.size();
        int numCols = lines.get(0).split(",").length;

        INDArray data = Nd4j.create(numRows, numCols);

        for (int i = 0; i < numRows; i++) {
            String[] values = lines.get(i).split(",");

            for (int j = 0; j < numCols; j++) {
                double val = Double.parseDouble(values[j]);
                data.putScalar(i, j, val);
            }
        }
        return data;
    }
}
