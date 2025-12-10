const form = document.getElementById("predictForm");
const input = document.getElementById("fileInput");
const author = document.getElementById("author");
const preview = document.getElementById("img_preview");
const result = document.getElementById("result");


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


// Handler /predict
form.addEventListener("submit", async (e) => {
    e.preventDefault();

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
        const response = await fetch("/predict", {method: "POST", body: formData});

        // Przetwarzanie odpowiedzi
        const data = await response.json();
        // Poniższe do zmiany jak zacznie działać model
        result.textContent = `Nazwa pliku: ${data.filename}\nSzerokość: ${data.size.width}px\nWysokość: ${data.size.height}px\nAutor: ${data.author}`;
      } catch (err) {
        result.textContent = err.message;
      }
});