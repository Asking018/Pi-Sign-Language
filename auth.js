// Pi Frontend SDK: authenticate in Pi ecosystem (works in Pi Browser when app is declared on Developer Portal)
const scopes = ['payments', 'username', 'profile'];

function onIncompletePaymentFound(payment) {
    if (payment) console.log('Incomplete payment found:', payment);
}

async function authenticateUser() {
    const connectButton = document.getElementById('connect-pi');
    const userInfo = document.getElementById('user-info');
    const usernameEl = document.getElementById('username');
    if (connectButton) connectButton.disabled = true;
    try {
        const auth = await Pi.authenticate(scopes, onIncompletePaymentFound);
        if (usernameEl) usernameEl.textContent = auth.user?.username || 'Pi User';
        if (userInfo) userInfo.classList.remove('hidden');
        if (connectButton) connectButton.style.display = 'none';
        localStorage.setItem('pi_auth_token', auth.accessToken);
        if (auth.user) localStorage.setItem('pi_user_info', JSON.stringify(auth.user));
    } catch (error) {
        console.error('Pi authentication error:', error);
        alert('Could not connect to Pi. Open this app in Pi Browser and ensure the app is declared on the Developer Portal (develop.pi).');
    } finally {
        if (connectButton) connectButton.disabled = false;
    }
}

// Add event listeners when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, setting up authentication...');
    const connectButton = document.getElementById('connect-pi');
    if (connectButton) {
        console.log('Connect button found, adding click listener');
        connectButton.addEventListener('click', function(e) {
            console.log('Connect button clicked');
            e.preventDefault();
            authenticateUser();
        });
    } else {
        console.log('No connect button found');
    }

    const authToken = localStorage.getItem('pi_auth_token');
    const userInfo = document.getElementById('user-info');
    const connectButton = document.getElementById('connect-pi');
    const usernameEl = document.getElementById('username');
    if (authToken && userInfo && connectButton) {
        const stored = localStorage.getItem('pi_user_info');
        if (usernameEl && stored) try { usernameEl.textContent = JSON.parse(stored).username || 'Pi User'; } catch (_) {}
        userInfo.classList.remove('hidden');
        connectButton.style.display = 'none';
    }
}); 