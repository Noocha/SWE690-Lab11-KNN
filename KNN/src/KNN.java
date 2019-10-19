import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;

enum Type {
    iris,
    car,
    unknown
}

public class KNN {
    String trainingDataFile;
    String predictDataFile;
    double bestK = 0;
    Type type;

    public Type getType() {
        return type;
    }

    public void setType(Type type) {
        this.type = type;
    }

    public KNN(String trainingDataFile, String predictDataFile) {
        this.trainingDataFile = trainingDataFile;
        this.predictDataFile = predictDataFile;
    }



    public Instances getDataSet(String filename) {
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(new File(filename));
            Instances dataSet = loader.getDataSet();
            return dataSet;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void train() {
        Instances trainData = getDataSet(trainingDataFile);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        try {
            Classifier ibk = new IBk();
            double percentComparison = 0;
            double currentPercentComparison = 0;

            for (int i = 3; i < 20; i+=2) {
                ibk = new IBk(i);
                ibk.buildClassifier(trainData);
                System.out.println(ibk);

                Evaluation eval = new Evaluation(trainData);
                eval.evaluateModel(ibk, trainData);

                currentPercentComparison = eval.correct()/eval.incorrect();

                if (currentPercentComparison > percentComparison) {
                    percentComparison = currentPercentComparison;
                    bestK = i;
                }
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toClassDetailsString());
                System.out.println(eval.toMatrixString());
            }

            System.out.println("The Best K is " + bestK);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void predict() {
        Classifier ibk = new IBk((int)bestK);
        Instances trainingData = getDataSet(trainingDataFile);
        trainingData.setClassIndex(trainingData.numAttributes() - 1);
        try {
            ibk.buildClassifier(trainingData);
            System.out.println("Prediction");
            Instances predictData = getDataSet(predictDataFile);
            predictData.setClassIndex(predictData.numAttributes() - 1);
            Instance predicInstance;
            double answer;

            for (int i = 0; i < predictData.numInstances(); i++) {
                predicInstance = predictData.instance(i);
                answer = ibk.classifyInstance(predicInstance);
                System.out.println(answer);
                if (type == Type.car) {
                    System.out.println(answer == 0 ? "unacc" : answer == 1 ? "acc" : answer == 2 ? "good" : answer == 3 ? "vgood" : "");

                } else {
//                    unacc, acc, good, vgood
                    System.out.println(answer == 0 ? "Iris-setosa" : answer == 1 ? "Iris-versicolor" : answer == 2 ? "Iris-virginica" : "");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
