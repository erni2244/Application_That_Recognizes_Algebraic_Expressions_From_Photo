package PictureTransform;

import org.jetbrains.annotations.NotNull;

public class Digit_coordinate implements Comparable {

    private int x;
    private int y;
    private int h;
    private int w;
    private String name;
    private int symbol;
    private  String symbol_wlasciwy;
    public Digit_coordinate(int x, int y, int w, int h, String name) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.name = name;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getH() {
        return h;
    }

    public int getW() {
        return w;
    }

    public String getName() {
        return name;
    }

    public void setSymbol(int symbol) {
        this.symbol = symbol;
    }

    public int getSymbol() {
        return symbol;
    }

    public String getSymbol_wlasciwy() {
        return symbol_wlasciwy;
    }

    public void setSymbol_wlasciwy(String symbol_wlasciwy) {
        this.symbol_wlasciwy = symbol_wlasciwy;
    }

    @Override
    public int compareTo(@NotNull Object o) {
        int p1 = ((Digit_coordinate) o).getX();
        int p2 = (this).getX();
        return Integer.compare(p2, p1);
    }
}
