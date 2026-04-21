# NRT Pipeline    
Tutti i comandi vanno eseguiti dalla root del repository, se non diversamente indicato.

## Setup

### Python 3.11

Verificare di avere Python 3.11 installato:

**macOS / Linux:**
```bash
python3.11 --version
```

**Windows (PowerShell):**
```powershell
py -3.11 --version
```

Se il comando dà errore, Python 3.11 non è installato. Scaricarlo da:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

> Su Windows assicurarsi di spuntare **"Add Python to PATH"** durante l'installazione.

### Make

`make` è preinstallato su macOS e Linux. Su Windows va installato manualmente.

Verificare che sia disponibile:

```
make --version
```

**Solo se il comando non viene trovato (Windows):** aprire **PowerShell come amministratore** ed eseguire i comandi seguenti per installare Chocolatey e poi `make`. Al termine chiudere VS Code e il terminale e riaprirli — è necessario perché il PATH venga aggiornato.

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; `
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

```powershell
choco install make -y
```

### Ambiente virtuale e dipendenze

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make setup
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
make setup
```

## AWS CLI

La AWS CLI è necessaria per accedere a S3 e Bedrock. Verificare che sia installata aprendo un terminale (Terminal su macOS/Linux, PowerShell o CMD su Windows) ed eseguendo:

```
aws --version
```

Se il comando non viene trovato, installarla seguendo la guida ufficiale (Windows, macOS e Linux):
[https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### Credenziali AWS

**Una persona per team** deve creare le credenziali seguendo questi passi:

1. Accedere alla console AWS con l'account fornito dagli organizzatori
2. Cercare **IAM** nella barra di ricerca in alto e aprirlo
3. Nel menu a sinistra cliccare **Users** → **Create user**
4. Scegliere un nome utente (es. `team-XX-dev`) e cliccare **Next**
5. Selezionare **Attach policies directly**, cercare `AdministratorAccess`, spuntarla e cliccare **Next** → **Create user**
6. Cliccare sull'utente appena creato → tab **Security credentials**
7. Scorrere fino ad **Access keys** → **Create access key**
8. Selezionare **Command Line Interface (CLI)** → spuntare la conferma in fondo → **Next**
9. Il tag Description può essere lasciato vuoto → **Create access key**
10. **Copiare subito** `Access key` e `Secret access key` — non saranno più visibili dopo aver chiuso questa pagina
11. Cliccare **Done**

Una volta ottenute le chiavi, configurare la CLI sul proprio computer:

```
aws configure
```

Inserire i valori richiesti:

```
AWS Access Key ID:     <Access key copiata al passo 10>
AWS Secret Access Key: <Secret access key copiata al passo 10>
Default region name:   eu-west-1
Default output format: json
```

Le credenziali vengono salvate in `~/.aws/credentials`. Ogni membro del team deve eseguire `aws configure` sulla propria macchina usando le stesse chiavi. 
 
## Variabili d'ambiente
  
``` 
cp .env.example .env 
```
  
Aprire `.env` e sostituire `XX` con il numero del proprio team.

## Verifica configurazione 

Prima di iniziare, verificare che l'ambiente sia configurato correttamente:

```
make validate
```

## Test e qualità del codice

```
make test    # esegue i test con coverage
make lint    # verifica la qualità del codice
```

> Questi comandi eseguono **esattamente gli stessi controlli** usati dalla pipeline di valutazione — il punteggio che vedete in locale è quello che riceverete.
