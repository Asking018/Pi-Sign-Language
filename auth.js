document.addEventListener("DOMContentLoaded", function () {

    console.log("Initializing Pi authentication...");

    const connectButton = document.getElementById("connect-pi");
    const userInfo = document.getElementById("user-info");
    const usernameEl = document.getElementById("username");

    if (!connectButton) {
        console.error("Connect button not found");
        return;
    }

    // restore previous login if saved
    const storedUser = localStorage.getItem("pi_user_info");
    if (storedUser) {
        try {
            const user = JSON.parse(storedUser);
            if (usernameEl) usernameEl.textContent = user.username;
            if (userInfo) userInfo.classList.remove("hidden");

            connectButton.textContent = `Connected: ${user.username}`;
            connectButton.disabled = true;
            return;
        } catch (e) {
            console.error("Error reading stored user", e);
        }
    }

    async function connectPi() {
        try {

            if (typeof Pi === "undefined") {
                alert("Please open this app inside Pi Browser.");
                return;
            }

            connectButton.disabled = true;
            connectButton.textContent = "Connecting...";

            const scopes = ["username", "payments"];

            function onIncompletePaymentFound(payment) {}

            const auth = await Pi.authenticate(scopes, onIncompletePaymentFound);

            console.log("Pi authenticated:", auth);

            localStorage.setItem("pi_auth_token", auth.accessToken);
            localStorage.setItem("pi_user_info", JSON.stringify(auth.user));

            if (usernameEl) usernameEl.textContent = auth.user.username;
            if (userInfo) userInfo.classList.remove("hidden");

            connectButton.textContent = `Connected: ${auth.user.username}`;

        } catch (error) {

            console.error("Pi authentication failed:", error);
            connectButton.disabled = false;
            connectButton.textContent = "Connect Pi Account";

        }
    }

    connectButton.addEventListener("click", connectPi);

});
