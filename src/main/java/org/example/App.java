package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App 
{
    static Logger log = LoggerFactory.getLogger(App.class);

    private static int lines2Skip = 1;
    private static double percentageOfTrain = 0.8; //Entero de 1 a 100
    private static int windowSize = 7;
    private static int minibatchSize = 4; //Number of examples in each minibatch
    private static int labelColumn = 0;
    public static void main( String[] args ) throws IOException, InterruptedException {
        log.info("Hola");
        List<String> lines = Files.readAllLines(Paths.get("test.csv"));
        log.info("En archivo csv número total de líneas: " + lines.size() + "\t\t Lines to skip: " + lines2Skip);
        log.info("Percent of train: " + percentageOfTrain + "\t\tWindow Size: " + windowSize +  "\t\tMinibatch size: " + minibatchSize);
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
        System.out.println(dataTest);
        INDArray dataTrain = getIndArray(trainList);
        long[] shape2 = dataTrain.shape();
        System.out.println("INDArray train: " + Arrays.toString(shape2));

        List<DataSet> winTest = createWindows(dataTest, 0);
        List<DataSet> winTrain = createWindows(dataTrain, 0);

        System.out.println("la lista de los datasets de test tiene size = " + winTest.size());
        System.out.println(winTest.get(3).getLabels());
        System.out.println();
        System.out.println(winTest.get(3).getFeatures());

        System.out.println("la lista de los datasets de train tiene size = " + winTrain.size());
        System.out.println(winTrain.get(0).getLabels());
        System.out.println();
        System.out.println(winTrain.get(0).getFeatures());
    }

    private static List<DataSet> createWindows(INDArray data, int labelIndex) {
        List<DataSet> windows = new ArrayList<>();

        int rows = data.rows();
        System.out.println("createWindows -> Numero de rows: " + rows);
        for(int i = 0; i < rows - windowSize; i++) {

            INDArray featureWindow = data.get(NDArrayIndex.interval(i, i + windowSize), NDArrayIndex.all());

            System.out.println("-----------------------------------------------");
            System.out.println("Imprimir el INDArray (el intervalo):");
            System.out.println(featureWindow);
            long[] shape = featureWindow.shape();
            System.out.println("INDArray featureWindow: " + Arrays.toString(shape));
            System.out.println("-----------------------------------------------");



            INDArray labelWindow =  data.get(NDArrayIndex.interval(i+1, i + windowSize +1), NDArrayIndex.point(labelIndex));

            System.out.println("-----------------------------------------------");
            System.out.println("Imprimir el INDArray (el intervalo):");
            System.out.println(labelWindow);
            long[] shape3 = labelWindow.shape();
            System.out.println("INDArray labelWindow: " + Arrays.toString(shape3));
            System.out.println("-----------------------------------------------");


            INDArray features3d = featureWindow
                    .transpose()
                    .reshape(1, featureWindow.columns(), windowSize);

            System.out.println("-----------------------------------------------");
            System.out.println("-----------------------------------------------");
            long[] shape4 = features3d.shape();
            System.out.println("INDArray featureWindow: " + Arrays.toString(shape4));
            System.out.println("-----------------------------------------------");


            INDArray labels3d = labelWindow.reshape(1, 1, windowSize);

            System.out.println("-----------------------------------------------");
            System.out.println(labels3d);
            System.out.println("-----------------------------------------------");
            long[] shape5 = labels3d.shape();
            System.out.println("INDArray labelWindow: " + Arrays.toString(shape5));
            System.out.println("-----------------------------------------------");

            windows.add(new DataSet(features3d, labels3d));
        }


        return windows;
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
