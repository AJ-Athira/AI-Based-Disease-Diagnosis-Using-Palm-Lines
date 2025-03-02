document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("uploadForm").addEventListener("submit", function(event) {
        event.preventDefault();
        
        let imageFile = document.getElementById("imageInput").files[0];

        if (!imageFile) {
            alert("Please select an image.");
            return;
        }

        let formData = new FormData();
        formData.append("image", imageFile);

        // Show image preview
        let reader = new FileReader();
        reader.onload = function(e) {
            let previewImage = document.getElementById("previewImage");
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
        };
        reader.readAsDataURL(imageFile);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Response:", data);
            let resultElement = document.getElementById("result");
            if (data.error) {
                resultElement.innerText = "Error: " + data.error;
                resultElement.style.color = "red";
            } else {
                resultElement.innerHTML = `
                    <strong>Predicted Disease:</strong> ${data.predicted_disease} <br>
                    <strong>Bounding Box:</strong> (${data.bounding_box.xmin}, ${data.bounding_box.ymin}) to 
                                                 (${data.bounding_box.xmax}, ${data.bounding_box.ymax})`;
                resultElement.style.color = "green";
            }
        })
        .catch(error => {
            console.error("Error:", error);
            document.getElementById("result").innerText = "Error: Unable to get prediction.";
            document.getElementById("result").style.color = "red";
        });
    });
});
