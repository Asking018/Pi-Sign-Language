async function payWithPi(amount, memo, plan) {

    if (!window.Pi) {
        alert("Open in Pi Browser");
        return;
    }

    const paymentData = {
        amount: amount,
        memo: memo,
        metadata: { plan: plan }
    };

    const callbacks = {

        onReadyForServerApproval: function(paymentId) {

            fetch("/approve_payment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ paymentId })
            });

        },

        onReadyForServerCompletion: function(paymentId, txid) {

            fetch("/complete_payment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ paymentId, txid })
            });

        },

        onCancel: function(paymentId) {

            console.log("Payment cancelled", paymentId);

        },

        onError: function(error) {

            console.error("Payment error", error);

        }
    };

    Pi.createPayment(paymentData, callbacks);

}