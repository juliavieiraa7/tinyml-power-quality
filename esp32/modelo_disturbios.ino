// Classes: [harmonico, normal, sag, swell, transitorio]
// Acurácia (treino/validação): ~99

#include "modelo_disturbios.h"      // Array do modelo TFLite (modelo_disturbios[])
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

// Configurações do modelo
// Modelo: 9 entradas (features) e 5 saídas (classes)
#define NUMBER_OF_INPUTS    9
#define NUMBER_OF_OUTPUTS   5

// Tamanho da arena de tensores
#define TENSOR_ARENA_SIZE   (60 * 1024)

// Habilita (1) ou desabilita (0) impressões de debug das saídas da rede
#define DEBUG_SAIDAS 1

// Objeto TinyML (EloquentTinyML + TensorFlow)
Eloquent::TinyML::TensorFlow::TensorFlow<
    NUMBER_OF_INPUTS,
    NUMBER_OF_OUTPUTS,
    TENSOR_ARENA_SIZE
> modelo;

// Parâmetros de normalização (copiados do StandardScaler do Python)
// mean_ = [ 0.74144129  0.09480114  0.89680397  0.64141642  5.30723809 10.47945142 32.87580017  1.56206776  0.57376892]
const float FEATURE_MEAN[NUMBER_OF_INPUTS] = {
  0.74144129f,
  0.09480114f,
  0.89680397f,
  0.64141642f,
  5.30723809f,
  10.47945142f,
  32.87580017f,
  1.56206776f,
  0.57376892f
};

// scale_ = [ 0.09218128  0.08138755  0.22491769  0.13272362  5.97507436 12.5279684 40.14182822  0.78424685  0.16347384]
const float FEATURE_SCALE[NUMBER_OF_INPUTS] = {
  0.09218128f,
  0.08138755f,
  0.22491769f,
  0.13272362f,
  5.97507436f,
  12.52796840f,
  40.14182822f,
  0.78424685f,
  0.16347384f
};

// Nomes das classes (NA MESMA ORDEM da saída do modelo)
const char* CLASS_NAMES[] = {
  "harmonico",
  "normal",
  "sag",
  "swell",
  "transitorio"
};

// Variável adicionada: classe real (ground truth)
int classe_real = 0;

// SETUP
void setup() {
  Serial.begin(115200);
  delay(2000);                 // tempo pro monitor serial conectar
  randomSeed(esp_random());   // ESP32 tem gerador aleatório de hardware

  Serial.println();
  Serial.println("Iniciando modelo (TensorFlow Lite + EloquentTinyML)...");
  Serial.println();

  bool ok = modelo.begin(modelo_disturbios);

  if (!ok || !modelo.isOk()) {
    Serial.println("ERRO em modelo.begin()");
    Serial.print("   Mensagem: ");
    Serial.println(modelo.getErrorMessage());
    Serial.println();
    Serial.println("Travando em loop de debug (mensagem será");
    while (true) {
      Serial.println("ERRO em modelo.begin(), ver mensagem acima.");
      delay(2000);
    }
  }

  Serial.println("Modelo carregado com sucesso!");
  Serial.print("   Entradas: ");
  Serial.println(NUMBER_OF_INPUTS);
  Serial.print("   Saídas: ");
  Serial.println(NUMBER_OF_OUTPUTS);
  Serial.print("   Arena: ");
  Serial.print(TENSOR_ARENA_SIZE / 1024);
  Serial.println(" kB");
  Serial.println("----------------------------------------");
  Serial.println("Setup terminou, entrando no loop()");
  Serial.println();
}

// LOOP
void loop() {
  static int counter = 0;
  float features[NUMBER_OF_INPUTS];

  gerarDadosSimulados(features, counter);
  classe_real = counter % 5;

  counter++;

  String resultado = classificarDisturbio(features);

  Serial.println("ANÁLISE DO DISTÚRBIO:");
  Serial.print("   Características (originais): RMS=");
  Serial.print(features[0], 2);
  Serial.print(", THD=");
  Serial.print(features[4], 1);
  Serial.print("%");
  Serial.print(" -> ");
  Serial.println(resultado);

  Serial.print("   Classe correta (ground truth): ");
  Serial.println(CLASS_NAMES[classe_real]);

  int erro = (resultado.startsWith(CLASS_NAMES[classe_real])) ? 0 : 1;
  Serial.print("   Erro (0=acerto,1=erro): ");
  Serial.println(erro);

  Serial.println("----------------------------------------");
  Serial.println();

  delay(3000);
}

// Vetores "centrais" usados como referência para cada classe
const float BASE_FEATURES[5][NUMBER_OF_INPUTS] = {
  {0.8209f, 0.1614f, 1.0103f, 0.6456f, 8.0540f, 10.2260f, 33.3720f, 1.5189f, 0.7315f},
  {0.7071f, 0.0000f, 0.7071f, 0.7071f, 0.0000f, -0.0000f, 0.0000f, 0.9999f, 0.5000f},
  {0.6510f, 0.0997f, 0.7071f, 0.4268f, 3.9468f, 8.3514f, 24.2248f, 0.9999f, 0.4380f},
  {0.7759f, 0.1195f, 1.0377f, 0.7071f, 2.3702f, 4.9395f, 14.0551f, 1.5033f, 0.6216f},
  {0.7411f, 0.0865f, 0.9874f, 0.7067f, 12.2803f, 28.2767f, 90.0104f, 2.6429f, 0.5623f}
};

void gerarDadosSimulados(float* features, int counter) {
  int tipo = counter % 5;

  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    float base = BASE_FEATURES[tipo][i];
    float fator = 1.0f + ( (random(-5, 6)) / 100.0f );
    features[i] = base * fator;
  }
}

String classificarDisturbio(float* features) {
  float saida[NUMBER_OF_OUTPUTS];

  float input_norm[NUMBER_OF_INPUTS];

  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    input_norm[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
  }

#if DEBUG_SAIDAS
  Serial.println("   Features normalizadas:");
  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    Serial.print("      x_norm[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(input_norm[i], 4);
  }
#endif

  modelo.predict(input_norm, saida);

  if (!modelo.isOk()) {
    Serial.println("ERRO em modelo.predict()");
    Serial.print("   Mensagem: ");
    Serial.println(modelo.getErrorMessage());
    return String("ERRO NA INFERÊNCIA");
  }

#if DEBUG_SAIDAS
  Serial.println("   Saída da rede (logits/scores):");
  for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
    Serial.print("      ");
    Serial.print(CLASS_NAMES[i]);
    Serial.print(" = ");
    Serial.println(saida[i], 6);
  }
#endif

  int   classe_predita = 0;
  float maior_score    = saida[0];

  for (int i = 1; i < NUMBER_OF_OUTPUTS; i++) {
    if (saida[i] > maior_score) {
      maior_score    = saida[i];
      classe_predita = i;
    }
  }

  char buffer[80];
  snprintf(buffer, sizeof(buffer), "%s (logit ≈ %.4f)",
           CLASS_NAMES[classe_predita], maior_score);

  return String(buffer);
}
