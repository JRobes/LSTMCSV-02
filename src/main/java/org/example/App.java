package org.example;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws IOException, InterruptedException {
        System.out.println( "Hello World!" );
        File baseDir = new File("C:\\Users\\COTERENA\\Desktop\\CSV");
        // We are using a random number generator to randomize the order
        InputSplit inputSplit = new FileSplit(baseDir);
        int numLinesToSkip = 0; //Optional, allows us to skip header lines
        String delimiter = ","; //Comma-delimited files
        SequenceRecordReader reader = new CSVSequenceRecordReader(numLinesToSkip, delimiter);
        reader.initialize(inputSplit);
        int minibatchSize = 10; //Number of examples in each minibatch
        int labelIndex = 0; //Index of column that contains the label
        int numClasses = 5; //Number of classes (label categories)
        DataSetIterator iterator =
                new SequenceRecordReaderDataSetIterator(reader, minibatchSize, labelIndex, numClasses);
        System.out.println(iterator.);
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

        //myNetwork.fit(iterator);
    }
}
