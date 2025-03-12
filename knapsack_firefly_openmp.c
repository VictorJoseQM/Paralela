#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#ifdef _WIN32
// Implementação simples de rand_r para Windows
int rand_r(unsigned int *seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (unsigned int)(*seed / 65536) % 32768;
}
#endif

// Definição de constantes para o algoritmo Firefly
#define NUM_FIREFLIES 5000   // Número de vaga-lumes na população
#define MAX_GENERATIONS 3000 // Número máximo de gerações
#define ALPHA 0.2            // Fator de aleatoriedade
#define BETA0 1.0            // Atratividade inicial
#define GAMMA 1.0            // Coeficiente de absorção da luz
#define PENALTY_FACTOR 10    // Penalidade aplicada a soluções inviáveis

// Estrutura para representar um vaga-lume (solução)
typedef struct
{
    int *solution; // Vetor binário representando a solução
    int profit;    // Lucro total da solução
    int weight;    // Peso total da solução
} Firefly;

// Variáveis globais para armazenar os dados do problema
int capacity;          // Capacidade máxima da mochila
int *weights;          // Vetor de pesos dos itens
int *profits;          // Vetor de lucros dos itens
int *optimal_solution; // Solução ótima fornecida pelo benchmark
int num_items;         // Número total de itens

// Função para ler os dados de entrada a partir dos arquivos
void read_data(const char *capacity_file, const char *weights_file, const char *profits_file, const char *solution_file)
{
    FILE *file;

    // Lendo a capacidade da mochila
    file = fopen(capacity_file, "r");
    fscanf(file, "%d", &capacity);
    fclose(file);

    // Contar quantos itens há no arquivo de pesos para alocar memória corretamente
    file = fopen(weights_file, "r");
    int count = 0;
    int temp;
    while (fscanf(file, "%d", &temp) != EOF)
        count++;
    fclose(file);

    num_items = count; // Definir o número correto de itens

    // Alocar memória para os arrays
    weights = (int *)malloc(num_items * sizeof(int));
    profits = (int *)malloc(num_items * sizeof(int));
    optimal_solution = (int *)malloc(num_items * sizeof(int));

    // Ler os pesos novamente agora que sabemos o tamanho correto
    file = fopen(weights_file, "r");
    for (int i = 0; i < num_items; i++)
        fscanf(file, "%d", &weights[i]);
    fclose(file);

    // Ler os lucros dos itens
    file = fopen(profits_file, "r");
    for (int i = 0; i < num_items; i++)
        fscanf(file, "%d", &profits[i]);
    fclose(file);

    // Ler a solução ótima do benchmark
    file = fopen(solution_file, "r");
    for (int i = 0; i < num_items; i++)
        fscanf(file, "%d", &optimal_solution[i]);
    fclose(file);
}

// Função para calcular o lucro total de uma solução
int calculate_profit(int *solution)
{
    int total = 0;
    for (int i = 0; i < num_items; i++)
        if (solution[i])
            total += profits[i];
    return total;
}

// Função para calcular o peso total de uma solução
int calculate_weight(int *solution)
{
    int total = 0;
    for (int i = 0; i < num_items; i++)
        if (solution[i])
            total += weights[i];
    return total;
}

// Inicializa a população de vaga-lumes com soluções aleatórias
void initialize_fireflies(Firefly fireflies[NUM_FIREFLIES])
{
    for (int i = 0; i < NUM_FIREFLIES; i++)
    {
        fireflies[i].solution = (int *)malloc(num_items * sizeof(int));
        for (int j = 0; j < num_items; j++)
            fireflies[i].solution[j] = rand() % 2; // Inicializa aleatoriamente com 0 ou 1

        fireflies[i].weight = calculate_weight(fireflies[i].solution);
        fireflies[i].profit = (fireflies[i].weight > capacity) ? 0 : calculate_profit(fireflies[i].solution);
    }
}

// Versão thread-safe da função move_firefly
// Utiliza rand_r() com uma semente privada
void move_firefly_threadsafe(Firefly *firefly, Firefly *brighter_firefly, unsigned int *seed)
{
    for (int i = 0; i < num_items; i++)
    {
        double r = (double)rand_r(seed) / RAND_MAX; // Número aleatório entre 0 e 1 (thread-safe)
        double beta = BETA0 * exp(-GAMMA * r * r);  // Atratividade decai com a distância
        firefly->solution[i] += beta * (brighter_firefly->solution[i] - firefly->solution[i]) + ALPHA * (r - 0.5);

        // Garantir que os valores continuem binários (0 ou 1)
        if (firefly->solution[i] < 0)
            firefly->solution[i] = 0;
        if (firefly->solution[i] > 1)
            firefly->solution[i] = 1;
    }

    firefly->weight = calculate_weight(firefly->solution);
    if (firefly->weight > capacity)
        firefly->profit = calculate_profit(firefly->solution) - (firefly->weight - capacity) * PENALTY_FACTOR;
    else
        firefly->profit = calculate_profit(firefly->solution);
}

// Implementa o algoritmo Firefly para otimizar a solução da mochila
void firefly_algorithm(int run_number, int *best_profit, int *best_weight, int num_threads)
{
    Firefly fireflies[NUM_FIREFLIES];
    initialize_fireflies(fireflies);

    // Evolução da população ao longo das gerações
    for (int gen = 0; gen < MAX_GENERATIONS; gen++)
    {
        // Cria um snapshot dos lucros dos vaga-lumes para evitar inconsistências durante a atualização
        int profit_snapshot[NUM_FIREFLIES];
        for (int k = 0; k < NUM_FIREFLIES; k++)
            profit_snapshot[k] = fireflies[k].profit;

// Atualiza cada vaga-lume de forma paralela
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < NUM_FIREFLIES; i++)
        {
            // Inicializa uma semente privada para cada thread
            unsigned int seed = time(NULL) ^ (omp_get_thread_num() + i);
            for (int j = 0; j < NUM_FIREFLIES; j++)
            {
                if (profit_snapshot[j] > fireflies[i].profit)
                    move_firefly_threadsafe(&fireflies[i], &fireflies[j], &seed);
            }
        }
    }

    // Encontrar o melhor vaga-lume da geração
    int best_local_profit = 0;
    int best_local_weight = 0;
    for (int i = 0; i < NUM_FIREFLIES; i++)
    {
        if (fireflies[i].profit > best_local_profit)
        {
            best_local_profit = fireflies[i].profit;
            best_local_weight = fireflies[i].weight;
        }
    }

    // Exibir os resultados da execução atual
    printf("\nExecução %d:\n", run_number);
    printf("Melhor solução encontrada:\n");
    printf("Lucro: %d\n", best_local_profit);
    printf("Peso: %d\n", best_local_weight);

    printf("Solução ótima:\n");
    printf("Lucro: %d\n", calculate_profit(optimal_solution));
    printf("Peso: %d\n", calculate_weight(optimal_solution));

    // Atualizar os melhores valores globais (cada execução é separada)
    if (best_local_profit > *best_profit)
    {
        *best_profit = best_local_profit;
        *best_weight = best_local_weight;
    }

    // Liberar a memória dos vaga-lumes
    for (int i = 0; i < NUM_FIREFLIES; i++)
        free(fireflies[i].solution);
}

// Função principal para executar o algoritmo
int main(int argc, char *argv[])
{
    srand(time(NULL));

    // Determinar o número de threads a ser usado (padrão: 2)
    int num_threads = 2;
    if (argc > 1)
        num_threads = atoi(argv[1]);

    // Exibir o número de threads configurado
    printf("Usando %d threads\n", num_threads);

    // Ler os dados do problema a partir dos arquivos
    read_data("p07_c.txt", "p07_w.txt", "p07_p.txt", "p07_s.txt");

    int num_runs = 3;
    double total_time = 0;
    int best_overall_profit = 0;
    int best_overall_weight = 0;

    // Executar o algoritmo múltiplas vezes para avaliar desempenho
    for (int i = 0; i < num_runs; i++)
    {
        clock_t start = clock();
        firefly_algorithm(i + 1, &best_overall_profit, &best_overall_weight, num_threads);
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        total_time += time_spent;
        printf("Tempo de execução: %f segundos\n", time_spent);
    }

    // Exibir um resumo das execuções
    printf("\nResumo das execuções:\n");
    printf("Melhor lucro encontrado: %d\n", best_overall_profit);
    printf("Peso correspondente: %d\n", best_overall_weight);
    printf("Tempo total de execução: %f segundos\n", total_time);

    free(weights);
    free(profits);
    free(optimal_solution);

    return 0;
}
