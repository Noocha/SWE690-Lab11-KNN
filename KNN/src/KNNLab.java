public class KNNLab {
    public static void main(String[] args) {

        String trainingDataFile = "src/irisTraining.arff";
        String predictDataFile = "src/irisPredict.arff";
        KNN model = new KNN(trainingDataFile, predictDataFile);
        model.setType(Type.iris);
        model.train();
        model.predict();

        String carTrainingDataFile = "src/carTraining.arff";
        String carTpredictDataFile = "src/carPredict.arff";
        KNN cmodel = new KNN(carTrainingDataFile, carTpredictDataFile);
        cmodel.setType(Type.car);
        cmodel.train();
        cmodel.predict();
    }
}
