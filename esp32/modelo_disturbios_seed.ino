// Classificador de Distúrbios - Modelo treinado na SEED-PQD
// Classes: [harmonico, normal, sag, swell, transitorio]

#include "modelo_disturbios_seed.h"  
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

#define NUMBER_OF_INPUTS    9
#define NUMBER_OF_OUTPUTS   5
#define TENSOR_ARENA_SIZE   (60 * 1024)

// Debug
#define DEBUG_SAIDAS 1

Eloquent::TinyML::TensorFlow::TensorFlow<
    NUMBER_OF_INPUTS,
    NUMBER_OF_OUTPUTS,
    TENSOR_ARENA_SIZE
> modelo;

// Parâmetros de normalização da base SEED-PQD

const float FEATURE_MEAN[NUMBER_OF_INPUTS] = {
  0.676182727f,
  0.0175993061f,
  0.705323873f,
  0.646414895f,
  731.438254f,
  711.403631f,
  2627.44471f,
  1.07024088f,
  0.22229869f
};

const float FEATURE_SCALE[NUMBER_OF_INPUTS] = {
  0.277409474f,
  0.0123819751f,
  0.28833951f,
  0.269158981f,
  571.951315f,
  1136.12935f,
  3885.37399f,
  0.501305976f,
  0.245956382f
};


// Nomes das classes

const char* CLASS_NAMES[] = {
  "harmonico",
  "normal",
  "sag",
  "swell",
  "transitorio"
};

int classe_real = 0;


// Setup

void setup() {
  Serial.begin(115200);
  delay(2000);
  randomSeed(esp_random());

  Serial.println();
  Serial.println("Iniciando modelo (TinyML + TensorFlow Lite) - SEED-PQD...");
  Serial.println();

  bool ok = modelo.begin(modelo_disturbios_seed);

  if (!ok || !modelo.isOk()) {
    Serial.println("ERRO em modelo.begin()");
    Serial.print("Mensagem: ");
    Serial.println(modelo.getErrorMessage());
    while (true) {
      Serial.println("Travado por erro no modelo.");
      delay(2000);
    }
  }

  Serial.println("Modelo carregado com sucesso!");
  Serial.print("Entradas: ");  Serial.println(NUMBER_OF_INPUTS);
  Serial.print("Saidas: ");    Serial.println(NUMBER_OF_OUTPUTS);
  Serial.print("Arena: ");     Serial.print(TENSOR_ARENA_SIZE / 1024); Serial.println(" kB");
  Serial.println("----------------------------------------");
}


//Loop

void loop() {
  static int counter = 0;
  float features[NUMBER_OF_INPUTS];

  gerarDadosSimulados(features, counter);
  classe_real = counter % 5;
  counter++;

  String resultado = classificarDisturbio(features);

  Serial.println("ANÁLISE DO DISTÚRBIO:");
  Serial.print("   Características (originais): RMS_mean=");
  Serial.print(features[0], 4);
  Serial.print(", THD_feature_raw=");
  Serial.print(features[4], 2);
  Serial.print(" -> ");
  Serial.println(resultado);

  float thd_z = (features[4] - FEATURE_MEAN[4]) / FEATURE_SCALE[4];
  Serial.print("   THD_z (normalizado) = ");
  Serial.println(thd_z, 4);

  Serial.print("   Classe correta (ground truth): ");
  Serial.println(CLASS_NAMES[classe_real]);

  int erro = resultado.startsWith(CLASS_NAMES[classe_real]) ? 0 : 1;
  Serial.print("   Erro (0=acerto,1=erro): ");
  Serial.println(erro);

  Serial.println("----------------------------------------");
  Serial.println();

  delay(3000);
}


//Vetores médios por classe

const float BASE_FEATURES[5][NUMBER_OF_INPUTS] = {


  // harmônico
  {
    0.7179f,
    0.0272f,
    0.7618f,
    0.6671f,
    921.3807f,
    956.8025f,
    3469.2947f,
    1.3126f,
    0.5136f
  },


  // normal
  {
    0.6364f,
    0.0000f,
    0.6364f,
    0.6364f,
    0.0000f,
    0.0000f,
    0.0000f,
    0.6364f,
    0.0000f
  },


  // sag
  {
    0.4812f,
    0.0131f,
    0.5029f,
    0.4596f,
    841.4310f,
    744.9097f,
    2790.9543f,
    0.6364f,
    0.0377f
  },

  // swell
  {
    0.8171f,
    0.0199f,
    0.8505f,
    0.7871f,
    876.7292f,
    707.7319f,
    2717.1852f,
    1.1451f,
    0.0293f
  },


  // transitorio
  {
    0.7282f,
    0.0278f,
    0.7750f,
    0.6818f,
    1017.6503f,
    1147.5741f,
    4159.7894f,
    1.6207f,
    0.5309f
  }
};


//Geração de dados simulados
void gerarDadosSimulados(float* features, int counter) {
  int tipo = counter % 5;

  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    float base = BASE_FEATURES[tipo][i];
    float fator = 1.0f + ((random(-5, 6)) / 100.0f);   // ±5%
    features[i] = base * fator;
  }
}

// Classificação (inferência TFLite)
String classificarDisturbio(float* features) {
  float saida[NUMBER_OF_OUTPUTS];
  float input_norm[NUMBER_OF_INPUTS];

  // Normalização SEED-PQD
  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    input_norm[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
  }

#if DEBUG_SAIDAS
  Serial.println("   Features normalizadas:");
  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    Serial.print("      x_norm["); Serial.print(i); Serial.print("] = ");
    Serial.println(input_norm[i], 4);
  }
#endif

  modelo.predict(input_norm, saida);

  if (!modelo.isOk()) {
    Serial.println("ERRO em modelo.predict()");
    Serial.print("Mensagem: ");
    Serial.println(modelo.getErrorMessage());
    return String("ERRO");
  }

#if DEBUG_SAIDAS
  Serial.println("   Saídas (logits):");
  for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
    Serial.print("      ");
    Serial.print(CLASS_NAMES[i]);
    Serial.print(" = ");
    Serial.println(saida[i], 6);
  }
#endif

  int classe_predita = 0;
  float maior_score = saida[0];

  for (int i = 1; i < NUMBER_OF_OUTPUTS; i++) {
    if (saida[i] > maior_score) {
      maior_score = saida[i];
      classe_predita = i;
    }
  }

  char buffer[80];
  snprintf(buffer, sizeof(buffer), "%s (logit ≈ %.4f)",
           CLASS_NAMES[classe_predita], maior_score);

  return String(buffer);
}
