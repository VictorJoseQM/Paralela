#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#ifdef _WIN32
// Implementação simples de rand_r para Windows
int rand_r(unsigned int *seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (unsigned int)(*seed / 65536) % 32768;
}
#endif

// Definição de constantes para o algoritmo Firefly
#define NUM_FIREFLIES 5000     // Número total de vaga-lumes
#define MAX_GENERATIONS 3000   // Número máximo de gerações
#define ALPHA 0.2              // Fator de aleatoriedade
#define BETA0 1.0              // Atratividade inicial
#define GAMMA 1.0              // Coeficiente de absorção da luz
#define PENALTY_FACTOR 10      // Penalidade aplicada a soluções inviáveis
#define MIGRATION_INTERVAL 100 // Intervalo de migração

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

void read_data(const char *capacity_file, const char *weights_file,
               const char *profits_file, const char *solution_file)
{
    FILE *file;

    file = fopen(capacity_file, "r");
    fscanf(file, "%d", &capacity);
    fclose(file);

    file = fopen(weights_file, "r");
    int count = 0;
    while (fscanf(file, "%*d") != EOF)
        count++;
    fclose(file);

    num_items = count;
    weights = (int *)malloc(num_items * sizeof(int));
    profits = (int *)malloc(num_items * sizeof(int));
    optimal_solution = (int *)malloc(num_items * sizeof(int));

    file = fopen(weights_file, "r");
    for (int i = 0; i < num_items; i++)
        fscanf(file, "%d", &weights[i]);
    fclose(file);

    file = fopen(profits_file, "r");
    for (int i = 0; i < num_items; i++)
        fscanf(file, "%d", &profits[i]);
    fclose(file);

    file = fopen(solution_file, "r");
    for (int i = 0; i < num_items; i++)
        fscanf(file, "%d", &optimal_solution[i]);
    fclose(file);
}

int calculate_profit(int *solution)
{
    int total = 0;
    for (int i = 0; i < num_items; i++)
        if (solution[i])
            total += profits[i];
    return total;
}

int calculate_weight(int *solution)
{
    int total = 0;
    for (int i = 0; i < num_items; i++)
        if (solution[i])
            total += weights[i];
    return total;
}

void move_firefly(Firefly *firefly, Firefly *brighter, unsigned int *seed)
{
    for (int i = 0; i < num_items; i++)
    {
        double r = (double)rand_r(seed) / RAND_MAX;
        double beta = BETA0 * exp(-GAMMA * r * r);
        double movement = beta * (brighter->solution[i] - firefly->solution[i]) + ALPHA * (r - 0.5);
        firefly->solution[i] = (int)(firefly->solution[i] + movement);
        firefly->solution[i] = firefly->solution[i] < 0 ? 0 : firefly->solution[i] > 1 ? 1
                                                                                       : firefly->solution[i];
    }
    firefly->weight = calculate_weight(firefly->solution);
    if (firefly->weight > capacity)
    {
        firefly->profit = calculate_profit(firefly->solution) - (firefly->weight - capacity) * PENALTY_FACTOR;
    }
    else
    {
        firefly->profit = calculate_profit(firefly->solution);
    }
}

void run_firefly_algorithm(int rank, int size, int run_number, double *time_spent)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Divisão de vaga-lumes entre processos
    int local_count = NUM_FIREFLIES / size;
    int remainder = NUM_FIREFLIES % size;
    if (rank < remainder)
        local_count++;

    Firefly *local_fireflies = (Firefly *)malloc(local_count * sizeof(Firefly));
    unsigned int seed = (unsigned int)time(NULL) + rank + run_number * 1000;

    // Inicialização local
    for (int i = 0; i < local_count; i++)
    {
        local_fireflies[i].solution = (int *)malloc(num_items * sizeof(int));
        for (int j = 0; j < num_items; j++)
        {
            local_fireflies[i].solution[j] = rand_r(&seed) % 2;
        }
        local_fireflies[i].weight = calculate_weight(local_fireflies[i].solution);
        local_fireflies[i].profit = (local_fireflies[i].weight > capacity) ? 0 : calculate_profit(local_fireflies[i].solution);
    }

    Firefly *external_bests = NULL;
    int migration_size = num_items + 2;
    int *send_buf = (int *)malloc(migration_size * sizeof(int));
    int *recv_buf = NULL;

    for (int gen = 0; gen < MAX_GENERATIONS; gen++)
    {
        // Atualização local
        for (int i = 0; i < local_count; i++)
        {
            for (int j = 0; j < local_count; j++)
            {
                if (local_fireflies[j].profit > local_fireflies[i].profit)
                {
                    move_firefly(&local_fireflies[i], &local_fireflies[j], &seed);
                }
            }
            if (external_bests)
            {
                for (int p = 0; p < size; p++)
                {
                    if (external_bests[p].profit > local_fireflies[i].profit)
                    {
                        move_firefly(&local_fireflies[i], &external_bests[p], &seed);
                    }
                }
            }
        }

        // Migração
        if (gen % MIGRATION_INTERVAL == 0)
        {
            // Encontrar melhor local
            int best_idx = 0;
            for (int i = 1; i < local_count; i++)
            {
                if (local_fireflies[i].profit > local_fireflies[best_idx].profit)
                {
                    best_idx = i;
                }
            }

            // Preparar buffer de envio
            for (int i = 0; i < num_items; i++)
            {
                send_buf[i] = local_fireflies[best_idx].solution[i];
            }
            send_buf[num_items] = local_fireflies[best_idx].profit;
            send_buf[num_items + 1] = local_fireflies[best_idx].weight;

            // Coletar melhores de todos os processos
            if (recv_buf == NULL)
            {
                recv_buf = (int *)malloc(size * migration_size * sizeof(int));
            }
            MPI_Allgather(send_buf, migration_size, MPI_INT, recv_buf, migration_size, MPI_INT, MPI_COMM_WORLD);

            // Atualizar melhores externos
            if (external_bests == NULL)
            {
                external_bests = (Firefly *)malloc(size * sizeof(Firefly));
                for (int p = 0; p < size; p++)
                {
                    external_bests[p].solution = (int *)malloc(num_items * sizeof(int));
                }
            }
            for (int p = 0; p < size; p++)
            {
                for (int i = 0; i < num_items; i++)
                {
                    external_bests[p].solution[i] = recv_buf[p * migration_size + i];
                }
                external_bests[p].profit = recv_buf[p * migration_size + num_items];
                external_bests[p].weight = recv_buf[p * migration_size + num_items + 1];
            }
        }
    }

    // Encontrar melhor local
    int best_local = 0;
    for (int i = 1; i < local_count; i++)
    {
        if (local_fireflies[i].profit > local_fireflies[best_local].profit)
        {
            best_local = i;
        }
    }

    // Coletar melhores globais
    int *global_profits = NULL;
    int *global_solutions = NULL;
    if (rank == 0)
    {
        global_profits = (int *)malloc(size * sizeof(int));
        global_solutions = (int *)malloc(size * num_items * sizeof(int));
    }

    MPI_Gather(&local_fireflies[best_local].profit, 1, MPI_INT, global_profits, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_fireflies[best_local].solution, num_items, MPI_INT,
               global_solutions, num_items, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcular tempo
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    *time_spent = end_time - start_time;

    // Liberar memória
    for (int i = 0; i < local_count; i++)
    {
        free(local_fireflies[i].solution);
    }
    free(local_fireflies);
    free(send_buf);
    free(recv_buf);
    if (external_bests)
    {
        for (int p = 0; p < size; p++)
        {
            free(external_bests[p].solution);
        }
        free(external_bests);
    }

    // Resultado final
    if (rank == 0)
    {
        int best_idx = 0;
        for (int p = 1; p < size; p++)
        {
            if (global_profits[p] > global_profits[best_idx])
            {
                best_idx = p;
            }
        }
        int best_profit = global_profits[best_idx];
        int *best_solution = &global_solutions[best_idx * num_items];
        int best_weight = calculate_weight(best_solution);

        printf("\nExecução %d:\n", run_number + 1);
        printf("Melhor solução encontrada:\n");
        printf("Lucro: %d\n", best_profit);
        printf("Peso: %d\n", best_weight);
        printf("Solução ótima:\n");
        printf("Lucro: %d\n", calculate_profit(optimal_solution));
        printf("Peso: %d\n", calculate_weight(optimal_solution));

        free(global_profits);
        free(global_solutions);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ler dados apenas no rank 0 e broadcast
    if (rank == 0)
        read_data("p07_c.txt", "p07_w.txt", "p07_p.txt", "p07_s.txt");

    MPI_Bcast(&capacity, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        weights = (int *)malloc(num_items * sizeof(int));
        profits = (int *)malloc(num_items * sizeof(int));
        optimal_solution = (int *)malloc(num_items * sizeof(int));
    }

    MPI_Bcast(weights, num_items, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(profits, num_items, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(optimal_solution, num_items, MPI_INT, 0, MPI_COMM_WORLD);

    int num_runs = 3;
    double total_time = 0;
    int best_profit = 0;

    for (int run = 0; run < num_runs; run++)
    {
        double time;
        run_firefly_algorithm(rank, size, run, &time);

        if (rank == 0)
        {
            total_time += time;
            printf("Tempo da execução %d: %.2f segundos\n", run + 1, time);
        }
    }

    if (rank == 0)
    {
        printf("\nResumo das execuções:\n");
        printf("Tempo total: %.2f segundos\n", total_time);
    }

    free(weights);
    free(profits);
    free(optimal_solution);
    MPI_Finalize();
    return 0;
}