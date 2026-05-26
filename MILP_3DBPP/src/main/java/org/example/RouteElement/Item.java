package org.example.RouteElement;

public class Item {
    public int id;
    public double x;
    public double y;
    public double z;
    public double weight;
    public int l; // length
    public int h; // height
    public int w; // width
    public int orientation;  //0-5 6directions
    public int visit_order = -1; // the order in which the item was loaded
    public int fragile = 0;  // 0: not fragile, 1: fragile

    public Item(int id, int x, int y, int z, int length, int width, int height, double mass, int orientation, int visit_order, int fragile){
        this.id = id;
        this.x = x;
        this.y = y;
        this.z = z;
        this.weight = mass;
        this.l = length;
        this.w = width;
        this.h = height;
        this.orientation = orientation;
        this.visit_order = visit_order;
        this.fragile = fragile;
    }
    // 使用新的输入格式构造
    public Item(int id, double length, double width, double height, int visitOrder, int fragile) {
        this.l = (int) length;
        this.w = (int) width;
        this.h = (int) height;
        this.visit_order = visitOrder;
        this.fragile = fragile;
    }
    public int[] getDimensions(){
        if (this.orientation == 0) {
            return new int[]{this.l, this.w, this.h};
        }
        else if (this.orientation == 1) {
            return new int[]{this.w, this.l, this.h};
        }
        else if (this.orientation == 2) {
            return new int[]{this.h, this.l, this.w};
        }
        else if (this.orientation == 3) {
            return new int[]{this.h, this.w, this.l};
        }
        else if (this.orientation == 4) {
            return new int[]{this.w, this.h, this.l};
        }
        else if (this.orientation == 5) {
            return new int[]{this.l, this.h, this.w};
        }
        else {
            // 处理orientation值超出范围的情况
            System.out.println("错误: orientation值必须在0到5之间，当前值为: " + this.orientation);
            exception(); // 调用异常处理方法
            return null;
        }

    }

    public double getVolume(){
        int[] dimensions = this.getDimensions();
        return dimensions[0] * dimensions[1] * dimensions[2];
    }

    public boolean if_overlaps(Item other){
        int[] dimensions1 = this.getDimensions();
        int[] dimensions2 = other.getDimensions();
        int l1 = dimensions1[0], w1 = dimensions1[1], h1 = dimensions1[2];
        int l2 = dimensions2[0], w2 = dimensions2[1], h2 = dimensions2[2];

        if (this.x + l1 <= other.x || this.x >= other.x + l2 ||
                this.y + w1 <= other.y || this.y >= other.y + w2 ||
                this.z + h1 <= other.z || this.z >= other.z + h2) {
            return false;
        }
        return true;

    }

    private void exception() {
        // 在此处理异常，比如抛出一个异常或打印日志
        throw new IllegalArgumentException("出现错误");
    }

    @Override
    public String toString() {
        return String.format(
                "Item(id=%s, x=%s, y=%s, z=%s, weight=%s, length=%s, width=%s, height=%s, orientation=%s, visit_order=%s, fragile=%s)",
                this.id, this.x, this.y, this.z, this.weight, this.l, this.w, this.h, this.orientation, this.visit_order, this.fragile
        );
    }

}





