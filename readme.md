# Progetto di Analisi Multilingua SER: EMOVO e RAVDESS

Questo repository contiene il codice per effettuare analisi sui dataset **EMOVO** (italiano) e **RAVDESS** (inglese), utilizzando vari algoritmi di Machine Learning tra cui **SVM** (Support Vector Machine), **Decision Trees (DT)** e **Linear Discriminant Analysis (LDA)**. L'obiettivo è sviluppare sistemi di **Speech Emotion Recognition (SER)** che analizzano e riconoscono emozioni dal parlato.

## Istruzioni per l'esecuzione

1. **Clone del repository**  
   Clona il repository sul tuo sistema locale utilizzando il comando:
   ```bash
   git clone https://github.com/tuo-username/repository-ser-analysis.git
   ```

2. **Dataset**  
   Il codice richiede i dataset **EMOVO** e **RAVDESS** per funzionare. I dataset non sono inclusi in questo repository e devono essere scaricati manualmente da [questo link](https://pera.com).

3. **Posizionamento dei dataset**  
   Dopo aver scaricato i dataset, posiziona la cartella chiamata `datasets_raw` nella **root** del progetto. La struttura della cartella dovrebbe essere:
   ```
   repository-ser-analysis/
   ├── datasets_raw/
   ├── python/
   ├── main.py
   └── ...
   ```

4. **Esecuzione del codice**  
   Per avviare l'analisi, esegui lo script principale **main.py** dalla root del progetto, che si trova nella cartella `python`. Utilizza il seguente comando:
   ```bash
   python python/main.py
   ```

## Nota importante

Il codice **non funziona semplicemente** scaricando ed eseguendo `main.py`. È necessario **scaricare i dataset** dal link sopra indicato e inserirli correttamente nella cartella `datasets_raw` nella root del progetto. Solo dopo aver completato questa operazione, potrai eseguire correttamente l'analisi.

## Algoritmi Utilizzati

Gli algoritmi di Machine Learning implementati in questo progetto sono:
- **SVM (Support Vector Machine)**
- **Decision Trees (DT)**
- **Linear Discriminant Analysis (LDA)**

Questi modelli sono utilizzati per analizzare e classificare le emozioni presenti nei dataset audio forniti.

## Struttura del Codice

Il codice è organizzato come segue:
- **main.py**: script principale per eseguire l'intera pipeline di analisi.
- **/python**: contiene il codice di supporto per la classificazione e l'analisi dei dataset.
- **datasets_raw/**: cartella in cui dovrai inserire i dataset scaricati.

## Contatti

Per domande o suggerimenti, non esitare a contattare l'autore del progetto o a creare una **issue** su GitHub.

