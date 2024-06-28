function navigateTo(sectionId) {
    // Remove active class from all links
    document.querySelectorAll('.header-link').forEach(link => link.classList.remove('active'));

    // Add active class to the clicked link
    document.querySelector(`[href="#${sectionId}"]`).classList.add('active');

    // Hide all sections
    document.querySelectorAll('.content').forEach(section => section.style.display = 'none');

    // Show the selected section
    document.getElementById(sectionId).style.display = 'block';

    console.log(`Navigating to ${sectionId}`);
}

document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.header-link').forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent default behavior of anchor tag
            var sectionId = this.getAttribute('href').substring(1);
            navigateTo(sectionId);
        });
    });

    // Set initial active link to Home
    navigateTo('home');

    // Add an event listener for the "Start here" link
    document.querySelector('.v44_7').addEventListener('click', function () {
        navigateTo('foundyou');
    });
    // Add an event listener for the "Start here" link
    document.querySelector('.v44_5').addEventListener('click', function () {
        navigateTo('foundyou');
    });
});

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            const video = document.getElementById('video_feed');
            video.srcObject = stream;
            video.play();
        })
        .catch(err => console.error("Error accessing camera: ", err));
}

function captureImage() {
    const video = document.getElementById('video_feed');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        $.ajax({
            type: 'POST',
            url: '/upload',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.status === 'success') {
                    $('#captured_image').attr('src', response.image_path);
                    navigateTo('recommendation');
                } else {
                    console.error('Failed to capture image.');
                }
            },
            error: function(error) {
                console.error('Error capturing image:', error);
            }
        });
    }, 'image/jpeg');
}

document.addEventListener('DOMContentLoaded', function () {
    startCamera();
    document.getElementById('capture_btn').addEventListener('click', captureImage);
    document.getElementById('upload_btn').addEventListener('click', function() {
        var fileInput = document.getElementById('file_input');
        var file = fileInput.files[0];
        var formData = new FormData();
        formData.append('file', file);

        $.ajax({
            type: 'POST',
            url: '/upload',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.status === 'success') {
                    $('#captured_image').attr('src', response.image_path);
                    navigateTo('recommendation');
                } else {
                    console.error('Failed to upload image.');
                }
            },
            error: function(error) {
                console.error('Error uploading image:', error);
            }
        });
    });
});
