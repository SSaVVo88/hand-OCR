const form = document.getElementById("predictForm");
const input = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const preview = document.getElementById("img_preview");
const submitButton = document.getElementById("submitButton");
const result = document.getElementById("result");
const statusBox = document.getElementById("serverStatus");
const statusOverlay = document.getElementById("serverOverlay");

let serverOK = false;

const HEALTH_URL = "http://127.0.0.1:8000/health";
const PREDICT_URL = "http://127.0.0.1:8000/predict";
const REQUEST_TIMEOUT = 8000; // ms


// Button
uploadButton.addEventListener("click", () => {
    input.click();
});


// Miniaturka
input.addEventListener("change", () => {
    result.textContent = "";
    if (input.files.length === 1) {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = e => {
            preview.src = e.target.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);
        submitButton.style.display = "inline-block";
    } else {
        preview.style.display = "none";
        submitButton.style.display = "none";
    }
});


// Submit
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!serverOK) {
        result.textContent = "Server is offline.";
        return;
    }

    if (input.files.length !== 1) {
        result.textContent = "Upload one file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", input.files[0]);

    form.querySelector("button").disabled = true;
    result.textContent = "Processing...";

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

    try {
        const response = await fetch(PREDICT_URL, {
            method: "POST",
            body: formData,
            signal: controller.signal
        });

        const data = await response.json();

        if (!response.ok) {
            result.textContent = data.detail || data.message || "Server error.";
            return;
        }

        result.textContent =
            `Nazwa pliku: ${data.filename}
Szerokość: ${data.size.width}px
Wysokość: ${data.size.height}px`;

    } catch (err) {
        if (err.name === "AbortError") {
            result.textContent = "Server timeout.";
        } else {
            result.textContent = "Connection error.";
        }
    } finally {
        clearTimeout(timeout);
        form.querySelector("button").disabled = false;
        input.value = "";
        submitButton.style.display = "none";
    }
});

// Status
async function checkServer() {
    try {
        const res = await fetch(HEALTH_URL, { cache: "no-store" });
        if (!res.ok) throw new Error();

        serverOK = true;
        statusBox.style.background = "green";
        statusBox.textContent = "Online";
        statusOverlay.style.display = "none";
    } catch {
        serverOK = false;
        statusBox.style.background = "red";
        statusBox.textContent = "Offline";
        statusOverlay.style.display = "flex";
    }
}

checkServer();
setInterval(checkServer, 2000);