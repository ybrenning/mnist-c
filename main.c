#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define DIM 28

typedef struct Matrix {
    float **elems;
    int rows;
    int cols;
} matrix_t;

matrix_t *create_matrix(int rows, int cols)
{
    matrix_t *matrix = (matrix_t *) malloc(sizeof(matrix_t));
    if (matrix == NULL) {
        printf("Error while allocating matrix memory\n");
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;

    matrix->elems = (float **) malloc(sizeof(float *) * rows);
    if (matrix->elems == NULL) {
        printf("Error while allocating matrix rows\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix->elems[i] = malloc(sizeof(float) * cols);
        if (matrix->elems[i] == NULL) {
            printf("Error while allocating matrix cols\n");
            return 0;
        }
    }

    return matrix;
}

void free_matrix(matrix_t *matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->elems[i]);
    }

    free(matrix->elems);
    free(matrix);
}

void print_matrix(matrix_t *matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            if (matrix->elems[i][j]) {
                printf("%.4f  ", matrix->elems[i][j]);
            } else {
                printf("0.  ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

matrix_t *mul(matrix_t *a, matrix_t *b)
{
    if (a->cols != b->rows) {
        printf("Dimension error, matrices must be of size lxm * mxn");
        return NULL;
    }

    matrix_t *res = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->elems[i][k] * b->elems[k][j];
            }
            res->elems[i][j] = sum;
        }
    }

    return res;
}

matrix_t *sum(matrix_t *a, matrix_t *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Dimension mismatch, both matrices must be mxn\n");
        return NULL;
    }

    matrix_t *res = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->elems[i][j] = a->elems[i][j] + b->elems[i][j];
        }
    }

    return res;
}

matrix_t *fill(matrix_t *matrix, float num)
{
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->elems[i][j] = num;
        }
    }

    return matrix;
}

matrix_t *fill_rand(matrix_t *matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->elems[i][j] = (float) rand() / RAND_MAX;
        }
    }

    return matrix;
}

typedef struct Image {
    matrix_t *content;
    int label;
} img;

img **read_csv(const char *filename, int num_samples)
{
    img **images = (img **) malloc(sizeof(img *) * num_samples);

    FILE *fp = fopen(filename, "r");

    #define MAXCHAR 5120
    char row[MAXCHAR];

    fgets(row, MAXCHAR, fp);
    char *token = strtok(row, ",");

    int current_row = 0;
    while (!feof(fp)) {
        int x = 0, y = 0;

        images[current_row] = (img *) malloc(sizeof(img));

        fgets(row, MAXCHAR, fp);
        token = strtok(row, ",");

        images[current_row]->label = atoi(token);
        token = strtok(NULL, ",");
        images[current_row]->content = create_matrix(DIM, DIM);
        while (token) {
            if (y == 28) {
                y = 0;
                x++;
            }
            if (x == 28) {
                break;
                break;
            }

            images[current_row]->content->elems[x][y++] = (float) atoi(token);
            token = strtok(NULL, ",");
        }

        current_row++;
    }

    fclose(fp);
    return images;
}

void free_img(img *image)
{
    free_matrix(image->content);
    free(image);
}

void free_imgs(img **images, int num_samples)
{
    for (int i = 0; i < num_samples; i++) {
        free_img(images[i]);
    }

    free(images);
}

void print_img(img *image)
{
    printf("Label: %d\n\n", image->label);
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            printf("%.4f  ", image->content->elems[i][j]);
        }

        printf("\n");
    }
}

int main(void)
{
    srand(69);
    img **data = read_csv("data/mnist_train.csv", TRAIN_SIZE);

    // Print first sample
    print_img(data[0]);

    free_imgs(data, TEST_SIZE);
    return 0;
}
