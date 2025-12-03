const form = document.getElementById('predictForm');
const input = document.getElementById('text');
const result = document.getElementById('result');


/* Handler /predict */
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    /* w przyszłości zmodyfikować do obsługi jpg */
    const text = input.value
    try {
        /* Wysyłanie inputu użytkownika i oczekiwanie na odpowiedź serwera */
        const resp = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text})
        });

        /* Obsługa błędu */
        if (!resp.ok) {
            const txt = await resp.text();
            throw new Error(`Błąd serwera: ${resp.status} ${resp.statusText} — ${txt}`);
        }

        /* Odbiór i wyświetlanie odpowiedzi */
        const data = await resp.json();
        input.value = '';
        result.textContent = data.text_out ?? JSON.stringify(data, null, 2);
    } catch (err) {
    result.textContent = '-';
    }
});