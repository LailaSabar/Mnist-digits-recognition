package org.sid.digits_recog;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.evaluation.classification.Evaluation;

public class CnnModelMnist{
	public static void main(String[] args) {
		long seed=1234;
		double learningRate=0.001;
		long height=28;
		long width=28;
		long depth=1;
		int outputSize=3;
		MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
				.seed(seed)
				.updater(new Adam(learningRate))
				.list()
				.setInputType(InputType.convolutionalFlat(height, width, depth))
				 .layer(0,new ConvolutionLayer.Builder()
                         .kernelSize(3,3)
                         .nIn(depth)
                         .stride(1,1)
                         .nOut(20)
                         .activation(Activation.RELU).build())
                  .layer(1, new SubsamplingLayer.Builder()
                          .poolingType(SubsamplingLayer.PoolingType.MAX)
                          .kernelSize(2,2)
                          .stride(2,2)
                          .build())
                 .layer(2, new ConvolutionLayer.Builder(3,3)
                         .stride(1,1)
                         .nOut(50)
                         .activation(Activation.RELU)
                         .build())
                 .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                         .kernelSize(2,2)
                         .stride(2,2)
                         .build())
                 .layer(4, new DenseLayer.Builder()
                         .activation(Activation.RELU)
                         .nOut(500)
                         .build())
                 .layer(5,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                         .activation(Activation.SOFTMAX)
                         .nOut(outputSize)
                         .build())

            .build();
		
		 MultiLayerNetwork model=new MultiLayerNetwork(configuration);
		 model.init();
		 System.out.println("Model training");
		 
	     String path=System.getProperty("user.home")+"/animals";
		 File trainFile=new File(path+"/train");
		 FileSplit trainFileSplit=new FileSplit(trainFile, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
		 RecordReader recordReaderTrain=new ImageRecordReader(height,width,depth,new ParentPathLabelGenerator());
		 try {
			recordReaderTrain.initialize(trainFileSplit);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 int batchSize=1;
		 DataSetIterator dataSetIteratorTrain=new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, outputSize);
		 DataNormalization scaler=new ImagePreProcessingScaler(0,1);
		 dataSetIteratorTrain.setPreProcessor(scaler);
	
		
		 UIServer uiServer=UIServer.getInstance();
		 StatsStorage statsStorage=new InMemoryStatsStorage();
		 uiServer.attach(statsStorage);
		 model.setListeners(new StatsListener(statsStorage));
		
		 
		 int numEpoch=1;
		 for(int i=0;i<numEpoch;i++) {
			 model.fit(dataSetIteratorTrain);
		 }
		 
		 
		 System.out.println("testing model .....");
		 
		 File testFile=new File(path+"/test");
		 FileSplit testFileSplit=new FileSplit(testFile, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
		 RecordReader recordReaderTest=new ImageRecordReader(height,width,depth,new ParentPathLabelGenerator());
		 try {
			recordReaderTest.initialize(testFileSplit);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 DataSetIterator dataSetIteratorTest=new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, outputSize);
		 DataNormalization scalerTest=new ImagePreProcessingScaler(0,1);
		 dataSetIteratorTest.setPreProcessor(scalerTest);
		 Evaluation evaluation=new Evaluation();
		 
		 while (dataSetIteratorTest.hasNext()) {
			 DataSet dataSet=dataSetIteratorTest.next();
			 INDArray features=dataSet.getFeatures();
			 INDArray targetLabels=dataSet.getLabels();
			 
			 INDArray predictions=model.output(features);
			 evaluation.eval(predictions, targetLabels);
		 }
		 System.out.println(evaluation.stats());
		 /*
		 while (dataSetIteratorTrain.hasNext()) {
			 DataSet dataSet=dataSetIteratorTrain.next();
			 INDArray features=dataSet.getFeatures();
			 INDArray labels=dataSet.getLabels();
			 System.out.println(features.shapeInfoToString());
			 System.out.println(labels);
			 System.out.println("---------------------");
		 }
		 */
		
	}
}