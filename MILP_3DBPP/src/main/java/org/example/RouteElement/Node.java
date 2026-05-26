package org.example.RouteElement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Node {
    public int ID;
    public double x;
    public double y;
    public List<Item> demands = new ArrayList<Item>();
    public int R;                  //所属节点路线


    public Node(int id, double x, double y) {
        this.ID = id;
        this.x = x;
        this.y = y;
    }
    public double calDistance(Node node) {
        return Parameters.distanceMatrix[this.ID][node.ID];
    }

    @Override
    public int hashCode(){
        return Objects.hash(ID);
    }

    //重写equals方法，用于判断两个Node是否相等，只看ID是否相等，耗时短。
    @Override
    public boolean equals(Object obj) {
        // 检查引用是否相同
        if (this == obj) {
            return true;
        }
        // 检查是否是同一类型
        if (!(obj instanceof Node)) {
            return false;
        }
        // 转换为 Node 类型
        Node other = (Node) obj;
        // 比较 ID 是否相等
        return this.ID == other.ID;
    }
    // 深拷贝方法
    public Node clone() {
        Node clonedNode = new Node(this.ID, this.x, this.y);
        // 浅拷贝Item列表，直接复制引用
        clonedNode.demands = new ArrayList<>(this.demands);
        return clonedNode;
    }
    public double calWeight(){
        double weight = 0;
        for(Item item:demands){
            weight+=item.weight;
        }
        return weight;
    }


}
