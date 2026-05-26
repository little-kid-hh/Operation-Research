package org.example.RouteElement;

public class Vehicle {
    public int Mass_Capacity;
    public int CargoSpace_Length;
    public int CargoSpace_Width;
    public int CargoSpace_Height;

    // Constructor
    public Vehicle(int massCapacity, int cargoSpaceLength, int cargoSpaceWidth, int cargoSpaceHeight) {
        Mass_Capacity = massCapacity;
        CargoSpace_Length = cargoSpaceLength;
        CargoSpace_Width = cargoSpaceWidth;
        CargoSpace_Height = cargoSpaceHeight;
    }
    @Override
    public String toString() {
        return "RouteElement.Vehicle{" +
                "massCapacity=" + Mass_Capacity +
                ", cargoSpaceLength=" + CargoSpace_Length +
                ", cargoSpaceWidth=" + CargoSpace_Width +
                ", cargoSpaceHeight=" + CargoSpace_Height +
                '}';
    }
}
