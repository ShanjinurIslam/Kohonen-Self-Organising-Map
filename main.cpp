#include <iostream>  // std::cout
#include <algorithm> // std::shuffle
#include <array>     // std::array
#include <random>    // std::default_random_engine
#include <chrono>    // std::chrono::system_clock
#include <time.h>
#include <fstream>
#define MAX_X 32
#define MAX_Y 32
#define MAXINPUT 3

using namespace std;

struct color
{
    double r;
    double g;
    double b;

    void normalize()
    {
        double sum = sqrt(r * r + g * g + b * b);
        r /= sum;
        g /= sum;
        b /= sum;
    }
};

struct Index
{
    int x;
    int y;
};

vector<color> data;

//random index generation done!
vector<int> _build_iteration_indexes(int data_len, int num_iterations)
{
    vector<int> arr;
    for (int i = 0; i < num_iterations; i++)
    {
        arr.push_back(i % data_len);
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto gen = std::default_random_engine(seed);
    shuffle(arr.begin(), arr.end(), gen);
    return arr;
}

double asymptotic_decay(double learning_rate, int t, int max_iter)
{
    return learning_rate / (1 + (1.0 * t) / ((1.0 * max_iter) / 2));
}

class SOM
{
private:
    int x;                // x dimention of SOM
    int y;                // y dimention of SOM
    int input_len;        // Number of elements
    double sigma;         //Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
    double learning_rate; //initial learning rate
    double weights[MAX_X][MAX_Y][MAXINPUT];
    double s[MAX_X][MAX_Y][MAXINPUT];
    int activation_map[MAX_X][MAX_Y];
    double g[MAX_X][MAX_Y];
    int neighx[MAX_X];
    int neighy[MAX_Y];
    int winner_map[MAX_X][MAX_Y];

public:
    //decay_function = asymptotic_decay

    /*
        gaussian define korte hobe
    */

    SOM(int x, int y, int input_len, double sigma = 1.0, double learning_rate = .5)
    {
        this->x = x;
        this->y = y;
        this->input_len = input_len;
        this->sigma = sigma;
        this->learning_rate = learning_rate;
        srand(1);
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < input_len; k++)
                {
                    weights[i][j][k] = 1 / sqrt((rand() % 10) + 1);
                }
            }
        }

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                activation_map[i][j] = 0;
            }
        }

        for (int i = 0; i < x; i++)
        {
            neighx[i] = i;
        }
        for (int i = 0; i < y; i++)
        {
            neighy[i] = i;
        }
    }

    void neighborhood(Index winner, double sigma)
    {
        double ax[MAX_X];
        double ay[MAX_Y];
        double d = 2 * 3.1416 * sigma * sigma;
        for (int i = 0; i < x; i++)
        {
            ax[i] = exp(-pow(neighx[i] - winner.x, 2) / d);
        }

        for (int i = 0; i < y; i++)
        {
            ay[i] = exp(-pow(neighx[i] - winner.y, 2) / d);
        }

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                g[i][j] = ax[i] * ay[j];
            }
        }
    }

    void activate(color input)
    {
        double d[3];
        d[0] = input.r;
        d[1] = input.g;
        d[2] = input.b;

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < input_len; k++)
                {
                    s[i][j][k] = d[k] - weights[i][j][k];
                }
            }
        }

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                activation_map[i][j] = sqrt((s[i][j][0] * s[i][j][0]) + (s[i][j][1] * s[i][j][1]) + (s[i][j][2] * s[i][j][2]));
            }
        }
    }

    Index FindWinner(color input)
    {
        Index winner;
        activate(input);
        int min = 1e9;
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                if (activation_map[i][j] < min)
                {
                    min = activation_map[i][j];
                    winner.x = i;
                    winner.y = j;
                }
            }
        }
        return winner;
    }

    void update(color input, Index winner, int t, int num_iterations)
    {
        double d[3];
        d[0] = input.r;
        d[1] = input.g;
        d[2] = input.b;

        double eta = asymptotic_decay(learning_rate, t, num_iterations);
        double sig = asymptotic_decay(sigma, t, num_iterations);
        //learning_rate -= 0.0002;
        //sigma -= 0.0001;
        //cout<<learning_rate<<" "<<sigma<<endl ;

        neighborhood(winner, sig);

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < input_len; k++)
                {
                    s[i][j][k] = d[k] - weights[i][j][k];
                    weights[i][j][k] += eta * g[i][j] * (s[i][j][k]);
                }
            }
        }
    }

    void train(vector<color> data, int num_iterations)
    {
        vector<int> iterations = _build_iteration_indexes(data.size(), num_iterations);
        for (int t = 0; t < num_iterations; t++)
        {
            //printf("EPOCH %d\n",t);
            color input = data[t];
            Index winner = FindWinner(input);
            //cout<<winner.x<<" "<<winner.y<<endl ;
            update(input, winner, t, num_iterations);

            //printWeights();
        }
    }

    void printWeights()
    {
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < input_len; k++)
                {
                    cout << weights[i][j][k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }

    void WinnerMap()
    {
        ofstream out("out.txt");
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                winner_map[i][j] = 0;
            }
        }

        for (int i = 0; i < data.size(); i++)
        {
            Index win = FindWinner(data[i]);
            winner_map[win.x][win.y]++;
            out << weights[win.y][win.y][0] << " " << weights[win.x][win.y][1] << " " << weights[win.x][win.y][2] << endl;
        }

        /*for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                cout << winner_map[i][j] << " ";
            }
        }*/
    }

    void GenerateColorMap(char *Fname,char *Oname)
    {
        vector<color> d;
        ifstream in(Fname);
        double r, g, b;
        while (in >> r >> g >> b)
        {
            color x;
            x.r = r;
            x.g = g;
            x.b = b;
            d.push_back(x);
        }

        ofstream out(Oname);
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                winner_map[i][j] = 0;
            }
        }

        for (int i = 0; i < d.size(); i++)
        {
            Index win = FindWinner(d[i]);
            winner_map[win.x][win.y]++;
            out << weights[win.y][win.y][0] << " " << weights[win.x][win.y][1] << " " << weights[win.x][win.y][2] << endl;
        }

        int min = -1 ;
        Index index;
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                if(min<winner_map[i][j]){
                    min = winner_map[i][j] ;
                    index.x = i ;
                    index.y = j ;
                }
                cout << winner_map[i][j] << " ";
            }
            cout<<endl ;
        }

        cout<<"Winner Cluster : "<<index.x<<" "<<index.y<<endl ;
    }
};

void ReadInputs()
{
    ifstream in("input.txt");
    double r, g, b;
    while (in >> r >> g >> b)
    {
        color x;
        x.r = r;
        x.g = g;
        x.b = b;
        data.push_back(x);
    }
}

SOM som(8,8,3, 1, 0.2);
int main()
{
    ReadInputs();
    cout<<"30000 Iterations"<<endl;
    som.train(data, 30000);
    som.printWeights();
    
    return 0;
}
