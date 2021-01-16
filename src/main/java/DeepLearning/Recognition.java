package DeepLearning;

import PictureTransform.Digit_coordinate;
import org.apache.log4j.BasicConfigurator;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.SVM;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.opencv.core.CvType.CV_8U;

public class Recognition {

    private static final String RESOURCES_FOLDER_PATH = "src/main/resources/read_odp/";
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private String[] maska = {"0","1","2","3","4","5","6","7","8","9","+","-",".","|"};
    public Recognition() {
        BasicConfigurator.configure();
    }

    public String recognise1() throws IOException {
        String odpowiedz="";
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("Siec_cyfr"),false);
        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        INDArray input = Nd4j.create(new int[]{1, HEIGHT * WIDTH});
        int n = 0;
        INDArray array;
        File folder=new File(RESOURCES_FOLDER_PATH);
        File[] files = folder.listFiles();

        files = sortByNumber(files);

        for(int i=files.length-1;i>=0;i--)
        {
            INDArray img = nativeImageLoader.asRowVector(files[i]);
            scaler.transform(img);
            input.putRow(n, img);
            n++;
            array =  model.output(input);
            double max=0;
            int index=0;
            for (int j=0;j<array.columns();j++) {
                System.out.println(j+" = "+(array.getDouble(j)*100));
                if(max<=array.getDouble(j)) {
                    max=array.getDouble(j);
                    index=j;
                }
            }

            System.out.println("rozpoznano: "+index+" na "+files[i].getName());
            odpowiedz=odpowiedz+(index)+" ";
        }

        System.out.println("Odpowiedz: "+odpowiedz);
        return odpowiedz;
    }

    public String recognise(List<Digit_coordinate> liczby) throws IOException {
        String odpowiedz="";
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("Siec_cyfr"),false);
        int n=0;
        INDArray array;

        File folder=new File(RESOURCES_FOLDER_PATH);
        File[] files = folder.listFiles();
        //files = sortByNumber(files);
        //for(int i=files.length-1;i>=0;i--)
        for(Digit_coordinate liczba : liczby)
        {
            File file = new File(RESOURCES_FOLDER_PATH+liczba.getName());
            NativeImageLoader nativeImageLoader2 = new NativeImageLoader(HEIGHT, WIDTH,1);
            INDArray img = nativeImageLoader2.asMatrix(file);
            n++;
            array =  model.output(img);

            double max=0;
            int index=0;
            for (int j=0;j<array.columns();j++) {
                //System.out.println(j+" = "+(array.getDouble(j)*100));
                if(max<=array.getDouble(j)) {
                    max=array.getDouble(j);
                    index=j;
                }
            }
            System.out.println("rozpoznano: "+index+" na "+file.getName());
            liczba.setSymbol(index);
            odpowiedz=odpowiedz+(index)+" ";

        }
        odpowiedz = make_equation(liczby);
        System.out.println("Odpowiedz: = "+odpowiedz);
        //System.out.println("Odpowiedz: "+odpowiedz);
        return odpowiedz;
    }

    public String recogniseHaar(List<Digit_coordinate> liczby) throws IOException{
        String odpowiedz="";
        //MultiLayerNetwork model = MultiLayerNetwork.load(new File("Siec_cyfr"),false);
        SVM svm = SVM.load("SVMclasyfier.xml");
        Mat data_mat = new Mat(1,HEIGHT*WIDTH,CV_8U);


        File folder=new File(RESOURCES_FOLDER_PATH);
        for(Digit_coordinate liczba : liczby)
        {
            File file = new File(RESOURCES_FOLDER_PATH+liczba.getName());
            Mat img = Imgcodecs.imread(file.getPath());
            for (int w = 0; w < WIDTH; w++) {
                for (int h = 0; h < HEIGHT; h++) {
                    data_mat.put(0, w * 28 + h, img.get(w, h));
                }
            }
            float index;
            data_mat.convertTo(data_mat, CvType.CV_32FC1);
            index = svm.predict(data_mat);

            System.out.println("rozpoznano: "+index+" na "+file.getName());
            liczba.setSymbol((int) index);
            odpowiedz=odpowiedz+(index)+" ";

        }
        System.out.println("Odpowiedz: = "+make_equation(liczby));
        //System.out.println("Odpowiedz: "+odpowiedz);
        return odpowiedz;
    }

    public File[] sortByNumber(File[] files) {
        Arrays.sort(files, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                int n1 = extractNumber(o1.getName());
                int n2 = extractNumber(o2.getName());
                return n2 - n1;
            }

            private int extractNumber(String name) {
                int i = 0;
                try {
                    int s = name.indexOf('_')+1;
                    int e = name.lastIndexOf('.');
                    String number = name.substring(s, e);
                    i = Integer.parseInt(number);
                } catch(Exception e) {
                    i = 0;
                }
                return i;
            }
        });
        return files;
    }


    public String make_equation(List<Digit_coordinate> liczby){
        Collections.sort(liczby);
        String odp= "";
        int koniec=0;
        List<Digit_coordinate> structura=new ArrayList<>();
        for (Digit_coordinate liczba : liczby) {             //tutaj potem trzeba dopisac łączenie znaków podwujnych jak naprzykład dwukropek
            liczba.setSymbol_wlasciwy(maska[liczba.getSymbol()] + " ");
        }

        Digit_coordinate pop=null;
        for (int i=0;i<liczby.size();i++) {
            Digit_coordinate liczba=liczby.get(i);
            if(pop==null){
                pop=liczba;
                koniec=pop.getX()+pop.getW();
            }else {
                structura.add(pop);
                if(koniec<liczba.getX()){        //nie nachodza

                    odp+=fun1(structura);
                    koniec=liczba.getX()+liczba.getW();
                    structura.clear();
                }else{
                    if(koniec<liczba.getX()+liczba.getW())   //to gdy liczba w struktorze kończyu się puźniej niż obecnie sprawdzana
                        koniec=liczba.getX()+liczba.getW();
                }
                pop=liczba;
            }


            if(i==liczby.size()-1){ //to dla ostatniego
                structura.add(liczba);
                odp+=fun1(structura);
                structura.clear();
            }

        }




        return odp;
    }

    private String fun1( List<Digit_coordinate> list){
        if(list.size()==0)
            return " ";

        if(list.size()==1)
            return maska[list.get(0).getSymbol()]+"";

        if(list.size()==2){
            if((list.get(0).getSymbol()==12 || list.get(1).getSymbol()==12) && (list.get(0).getSymbol()==13 || list.get(1).getSymbol()==13) )
                return "! ";
            if((list.get(0).getSymbol()==12 && list.get(1).getSymbol()==12))
                return "/";
            else
                return "/ ";
        }


        Digit_coordinate kreska=list.get(0);
        for(Digit_coordinate digit_coordinate : list){
            if(kreska.getW()<digit_coordinate.getW())
                kreska=digit_coordinate;
        }

        list.remove(kreska);
        List<Digit_coordinate> gora = new ArrayList<>();
        List<Digit_coordinate> dol =new ArrayList<>();
        for(Digit_coordinate digit_coordinate : list){
            if(digit_coordinate.getY()+digit_coordinate.getH()<kreska.getY())
                gora.add(digit_coordinate);
            else
                dol.add(digit_coordinate);
        }

        return "("+make_equation(gora)+")/("+make_equation(dol)+")";
    }

}
