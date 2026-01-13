const form = document.getElementById("predictForm");
const input = document.getElementById("fileInput");
const author = document.getElementById("author");
const preview = document.getElementById("img_preview");
const result = document.getElementById("result");
const statusBox = document.getElementById("serverStatus");
const statusOverlay = document.getElementById("serverOverlay");

let serverOK = false;
let lastPing = null;

const HEALTH_URL = "http://127.0.0.1:8000/health";
const PREDICT_URL = "http://127.0.0.1:8000/predict";


// Miniaturka
input.addEventListener("change", () => {
    result.textContent = "";
    if (input.files.length === 1){
        const file = input.files[0];
        const reader = new FileReader();

        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = "block"};

        reader.readAsDataURL(file);
    } else{
        preview.src = "";
        preview.style.display = "none";
    }
});


// Handler submit
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Jeśli serwer nie działa
    if (!serverOK) {
        result.textContent = "Backend offline - uruchom serwer.";
        return;
    }

    if (input.files.length !== 1) {
        result.textContent = "Wybierz plik.";
        return;
    }
    if (!author.value) {
        result.textContent = "Wybierz kategorię.";
        return;
    }

    const formData = new FormData();
    formData.append("file", input.files[0]);
    formData.append("author", author.value);
    input.value = "";
    author.value = "";

    try {
        // Wysłanie pliku do Pythona
        const response = await fetch(PREDICT_URL, {method: "POST", body: formData});

        // Przetwarzanie odpowiedzi
        const data = await response.json();
        // Poniższe do zmiany jak zacznie działać model
        result.textContent = `Nazwa pliku: ${data.filename}\nSzerokość: ${data.size.width}px\nWysokość: ${data.size.height}px\nAutor: ${data.author}`;
      } catch (err) {
        result.textContent = err.message;
      }
});


// Sprawdzanie statusu
async function checkServer() {
    // Ping
    const t0 = performance.now();
    try {
        const res = await fetch(HEALTH_URL, {cache: "no-store"});
        if (!res.ok) throw new Error();

        // Ping
        const t1 = performance.now();
        lastPing = Math.round(t1 - t0);
        serverOK = true;

        statusBox.style.background = "green";
        statusBox.textContent = `Online (${lastPing} ms)`;
        statusOverlay.style.display = "none";

    } catch {
        serverOK = false;
        statusBox.style.background = "red";
        statusBox.textContent = "Offline - uruchom serwer";
        statusOverlay.style.display = "flex";
    }
}
checkServer();
setInterval(checkServer, 2000);