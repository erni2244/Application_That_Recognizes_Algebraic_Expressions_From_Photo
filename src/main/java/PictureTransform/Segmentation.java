package PictureTransform;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

public class Segmentation {

    private static final String RESOURCES_FOLDER_PATH = "src/main/resources/read_odp/";
    private Size size_element = new Size(28,28);
    private String file_patch;
    private Imgcodecs imageCodecs=new Imgcodecs();
    private Mat matrix;

    public Segmentation(String file_patch) {
        this.file_patch=file_patch;
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );

    }

    private void load_file_to_matrix(){
        try{
            matrix = Imgcodecs.imread(file_patch);
        }catch (Exception e){
            System.out.println("Error");
        }
    }

    public boolean check_matrix(){
        return matrix !=null;
    }


    public List<Digit_coordinate> separate(){
        load_file_to_matrix();
        if(check_matrix())
            return cut_digit();
        return null;
    }


    private List<Digit_coordinate> cut_digit(){
        clear_folder(RESOURCES_FOLDER_PATH);
        List<MatOfPoint> contours= new ArrayList<>();
        Mat hierarchy=new Mat();
        Mat odp=new Mat();
        int padding=5;
        boolean heck=true;
        Imgproc.cvtColor(matrix, odp, Imgproc.COLOR_BGR2GRAY);
        /**
         sprawdz kolor tła i obruć w razie potrzeby
         */
        Imgproc.threshold(odp, odp, 60, 255, THRESH_BINARY);
        List<Digit_coordinate> liczby=new ArrayList<>();

        Imgproc.findContours(odp,contours,hierarchy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

        matrix =odp;
        List<Point> points;
        File[] files=new File[contours.size()];
        System.out.println(""+contours.size());
        for(int i=0; i<contours.size();i++) {
            points = contours.get(i).toList();
            int x, y, h, w;
            x = (int) points.get(0).x;
            w = (int) points.get(0).x;
            y = (int) points.get(0).y;
            h = (int) points.get(0).y;
            for (Point point : points) {
                if (x > (int) point.x)
                    x = (int) point.x;
                if (w < (int) point.x)
                    w = (int) point.x;
                if (y > (int) point.y)
                    y = (int) point.y;
                if (h < (int) point.y)
                    h = (int) point.y;
            }
            /**
             sprawdz minimalnej wielkości kontorów (tymczasowo usunięte)
             */

            int paddingW=3;
            Rect rectCrop0 = new Rect(x - paddingW, y - paddingW, w - x + (2 * paddingW), h - y + (2 * paddingW));
            Mat krawedz = new Mat(matrix, rectCrop0);
            //imageCodecs.imwrite("ty" + i + ".png", krawedz);
            for (int xx=0;xx<krawedz.rows();xx++) {
                if (krawedz.get(xx, 0)[0] > 5 || krawedz.get(xx, krawedz.cols()-1)[0] > 5 ) {
                    heck=false;
                    break;
                }
            }
            for (int xx=0;xx<krawedz.cols();xx++) {
                if ( krawedz.get(0, xx)[0] > 5 || krawedz.get(krawedz.rows()-1, xx)[0] > 5) {
                    heck=false;
                    break;
                }
            }
            /*
            if(heck) {
                Rect rectCrop = new Rect(x - padding, y - padding, w - x + 2 * padding, h - y + 2 * padding);
                System.out.println("---"+rectCrop);
                Mat croppedImage = new Mat(matrix, rectCrop);
                Mat resizeImage = new Mat();
                Imgproc.resize(croppedImage, resizeImage, size_element);
                Imgproc.threshold(resizeImage, resizeImage, 125, 255, THRESH_BINARY);
                save_image(resizeImage, (x + ".png"));
            }
            */
            if(heck) {
                int roznica=(h - y + 2 * padding)-(w - x + 2 * padding);
                Digit_coordinate nowy = new Digit_coordinate(x , y , w - x, h - y ,(i + ".png"));
                //Rect rectCrop = new Rect(x - padding-roznica/2, y - padding, h - y + 2 * padding, h - y + 2 * padding);
                Rect rectCrop;
                if(roznica>0)
                    rectCrop = new Rect(x - padding-roznica/2, y - padding, h - y + 2 * padding, h - y + 2 * padding);
                else {
                    roznica=10;
                    rectCrop = new Rect(x - padding, y - padding, w - x + 2 * padding, h - y + 2 * padding+roznica);    //to jest warunek dla minusów bo one są szersze niż wyrzsze
                }
                System.out.println("---"+rectCrop);
                Mat croppedImage = new Mat(matrix, rectCrop);
                //save_image(croppedImage, (x + "---11.png"));
                Mat resizeImage = new Mat();
                Imgproc.resize(croppedImage, resizeImage, size_element);
                //save_image(resizeImage, (x + "---22.png"));
                Imgproc.threshold(resizeImage, resizeImage, 60, 255, THRESH_BINARY);
                save_image(resizeImage, (nowy.getName()));
                liczby.add(nowy);
            }
            heck=true;
        }

        return liczby;
    }





    private void save_image(Mat m,String s){
        String string =RESOURCES_FOLDER_PATH+s;
        try {
            imageCodecs.imwrite(string, m);
        }catch (Exception e){
            System.out.println("Error save image");
        }
    }

    private void clear_folder(String path){
        File folder=new File(path);
        File[] files = folder.listFiles();
        for(File f : files){
            f.delete();
        }


    }

//funkcja urzyta tylko raz do przerobienia bazy danych uczącej sieć

    public void movepicture(){
        String path_read="C:/Users/lasek/IdeaProjects/baza_danych/mnist_png2/training/"; //folder with data base
        File folder = new File(path_read);
        File[] digitFolders = folder.listFiles();
        int przesuniecie_x=-2;
        int przesuniecie_y=0;
        int count=0;
        File digitFolder = new File(path_read+"14/");
        //for (File digitFolder : digitFolders) {
            File[] imageFiles = digitFolder.listFiles();
            int nr = imageFiles.length;
            System.out.println("--"+nr);
            count=0;
            for (File imgFile : imageFiles) {
                if(count>509)
                    break;
                count++;
                Mat m = Imgcodecs.imread(imgFile.getPath()).clone();
                Mat odp = m.clone();
                for(int x=0;x<m.cols();x++)
                    for(int y=0;y<m.rows();y++) {
                        if((x-przesuniecie_x>0 && x-przesuniecie_x<m.cols()) && (y-przesuniecie_y>0 && y-przesuniecie_y<m.rows())){
                            odp.put(x,y,m.get(x-przesuniecie_x,y-przesuniecie_y));
                        }else {
                            odp.put(x,y, 0, 0, 0);
                        }
                    }
                System.out.println(digitFolder.getPath()+"\\"+digitFolder.getName()+"_"+nr+".png");
                    nr++;
                    Imgproc.cvtColor(odp,odp,Imgproc.COLOR_RGB2GRAY);
                Imgcodecs.imwrite(digitFolder.getPath()+"\\"+digitFolder.getName()+"_"+nr+".png",odp);
            }
        //}
    }

    private Mat f11(Mat matW){
        Mat resizeImage = new Mat();
        clear_folder(RESOURCES_FOLDER_PATH);
        List<MatOfPoint> contours= new ArrayList<>();
        Mat hierarchy=new Mat();
        Mat odp;
        int padding=10;
        boolean heck=true;
        //Imgproc.cvtColor(matW, odp, Imgproc.COLOR_BGR2GRAY);
        odp=matW;
        Imgproc.threshold(odp, odp, 60, 255, THRESH_BINARY);
        Imgproc.findContours(odp,contours,hierarchy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

        matW =odp;
        List<Point> points;
        File[] files=new File[contours.size()];
        for(int i=0; i<contours.size();i++) {
            points = contours.get(i).toList();
            int x, y, h, w;
            x = (int) points.get(0).x;
            w = (int) points.get(0).x;
            y = (int) points.get(0).y;
            h = (int) points.get(0).y;
            for (Point point : points) {
                if (x > (int) point.x)
                    x = (int) point.x;
                if (w < (int) point.x)
                    w = (int) point.x;
                if (y > (int) point.y)
                    y = (int) point.y;
                if (h < (int) point.y)
                    h = (int) point.y;
            }

            int paddingW=3;
            Rect rectCrop0 = new Rect(x - paddingW, y - paddingW, w - x + (2 * paddingW), h - y + (2 * paddingW));
            Mat krawedz = new Mat(matW, rectCrop0);
            for (int xx=0;xx<krawedz.rows();xx++) {
                if (krawedz.get(xx, 0)[0] > 5 || krawedz.get(xx, krawedz.cols()-1)[0] > 5 ) {
                    heck=false;
                    break;
                }
            }
            for (int xx=0;xx<krawedz.cols();xx++) {
                if ( krawedz.get(0, xx)[0] > 5 || krawedz.get(krawedz.rows()-1, xx)[0] > 5) {
                    heck=false;
                    break;
                }
            }

            if(heck) {
                int roznica=(h - y + 2 * padding)-(w - x + 2 * padding);
                Rect rectCrop = new Rect(x - padding-roznica/2, y - padding, h - y + 2 * padding, h - y + 2 * padding);
                Mat croppedImage = new Mat(matW, rectCrop);
                Imgproc.resize(croppedImage, resizeImage, size_element);
                Imgproc.threshold(resizeImage, resizeImage, 60, 255, THRESH_BINARY);

            }
            heck=true;
        }

    return resizeImage;
    }

    public void fff() {
        File folder = new File("C:/Users/lasek/IdeaProjects/baza_danych/mnist_png/training/");
        File[] digitFolders = folder.listFiles();
        String s;
        String p;
        for (File digitFolder : digitFolders) {
            File[] imageFiles = digitFolder.listFiles();

            for (File imgFile : imageFiles) {
                p=imgFile.getPath();
                Mat m = Imgcodecs.imread(p);
                s=imgFile.getName();
                imgFile.delete();


                for(int jj=0;jj<28;jj++){
                    for(int ii=0;ii<28;ii++){
                        Mat ma=new Mat(28,28,1);
                        double[] data= m.get(ii,jj);
                        if(data[0]>125 || data[1]>125 || data[1]>125){
                            data[0]=255;
                            data[1]=255;
                            data[2]=255;
                        }else {
                            data[0]=0;
                            data[1]=0;
                            data[2]=0;
                        }
                        m.put(ii,jj,data);
                    }
                }
                Imgproc.cvtColor(m, m, Imgproc.COLOR_RGB2GRAY);
                imageCodecs.imwrite(p, m);
            }
        }
    }

    public void make_grid(){
        int szer_line=2;
        int szer_grid=28;
        szer_grid*=2;

        String path = "C:/Users/lasek/Desktop/grid.png";
        Mat mat;
        mat = Imgcodecs.imread(path);
        int a=0;
        for (int i=0;i<=17;i++){
            for (int j=0;j<mat.cols();j++){
                for(int z=0;z<szer_line;z++){
                    double[] data= mat.get(i*szer_grid+a+z,j);
                    data[0]=255;
                    data[1]=255;
                    data[2]=255;
                    mat.put(i*szer_grid+a+z,j,data);
                }
            }
            a+=szer_line;
        }

        a=0;
        for (int i=0;i<=30;i++){
            for (int j=0;j<mat.rows();j++){
                for(int z=0;z<szer_line;z++){
                    double[] data= mat.get(j,i*szer_grid+a+z);
                    data[0]=255;
                    data[1]=255;
                    data[2]=255;
                    mat.put(j,i*szer_grid+a+z,data);
                }
            }
            a+=szer_line;
        }
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY);
        imageCodecs.imwrite(path, mat);
    }

    public void extract_grid(){
        String znak ="99";
        int szer_line=2;
        int szer_grid=28;
        szer_grid*=2;
        String path_file = "C:/Users/lasek/Desktop/"+znak+"/";
        String path = "C:/Users/lasek/Desktop/"+"a.png";
        Mat mat;
        mat = Imgcodecs.imread(path);
        int nazwa=0;
        for(int i=0;i<17;i++)
            for (int j=0;j<30;j++){
                Rect rectCrop = new Rect(szer_grid*j + szer_line*(j+1), szer_grid*i + szer_line*(i+1), szer_grid, szer_grid);
                Mat croppedImage = new Mat(mat, rectCrop);
                Mat resizeImage = new Mat();
                Imgproc.resize(croppedImage, resizeImage, size_element);
                Imgproc.threshold(resizeImage, resizeImage, 125, 255, THRESH_BINARY);
                Imgproc.cvtColor(resizeImage,resizeImage,Imgproc.COLOR_RGB2GRAY);
                Imgcodecs.imwrite((path_file+znak+"_"+nazwa + ".png"), resizeImage);
                nazwa++;
            }


    }

}
