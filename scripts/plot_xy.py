import argparse
import matplotlib.pyplot as plt

def read_data(file_path):
    x_values = []
    y_values = []

    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split())
            x_values.append(x)
            y_values.append(y)
    
    return x_values, y_values

def plot_data(x_values, y_values):
    plt.figure()
    plt.plot(x_values, y_values, marker='o')
    plt.title('Plot of x vs y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot x vs y from a file.')
    parser.add_argument('file_path', type=str, help='Path to the input file')
    args = parser.parse_args()

    x_values, y_values = read_data(args.file_path)
    plot_data(x_values, y_values)

if __name__ == '__main__':
    main()
