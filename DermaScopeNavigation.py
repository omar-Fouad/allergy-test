import streamlit as st
from streamlit_chat import message  # For creating a conversational layout
import random
from PIL import Image
import time
import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import io


def load_model(model_name: str, encoder_name: str, checkpoint_path: str, device: torch.device):
    """Load the segmentation model from a checkpoint."""
    model = getattr(smp, model_name)(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1).to(
        device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state'])
    model.eval()
    return model


def segment_image(model, image_path, transform, device):
    """Apply segmentation on a single image."""
    from PIL import Image  # Ensure PIL is imported
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).cpu().squeeze().numpy() > 0.5
    return image, pred_mask



def save_segmented_image(original_image, segmented_mask, output_path):
    """Save the original image alongside the segmented mask."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_mask, cmap="gray")
    plt.title("Segmented Mask")
    plt.axis("off")
    plt.savefig(output_path)  # Save to the specified output path
    plt.close()




sound_file = "beep.mp3"

def play_sound_html(sound_path):
    """
    Embeds an HTML5 audio player to play sound automatically.
    """
    sound_html = f"""
    <audio autoplay>
        <source src="data:audio/mpeg;base64,{sound_path}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

# Convert sound file to base64
def get_base64_sound(file_path):
    """
    Reads a sound file and converts it to a base64 string.
    """
    import base64
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode()
    
base64_sound = get_base64_sound(sound_file)

# Step titles and content
steps = [
    "Home/Welcome Page", "Verify Kit Contents", "Set Up Quanti-Wells", "Prepare Quanti-Tray",
    "Load Applicators", "Prepare Skin Test Area", "Apply Test", "Record and Analyze Results",
    "Manage Medication Interference", "Results Summary",
]

if "current_step" not in st.session_state:
    st.session_state.current_step = 0


# Mock AI function (you can expand this with actual image processing later)
def mock_ai_verification(uploaded_image):
    # Placeholder logic for image verification
    # Ideally, you could integrate real AI/ML-based verification here.
    return "All items are verified and present in the kit!"

# Mock function to simulate template matching
def mock_template_matching(image_path):
    # Simulate analysis results
    matched = random.choice([True, False])
    details = {
        "status": "Correctly Placed" if matched else "Misaligned",
        "suggestion": "No adjustment needed." if matched else "Reposition stickers for accuracy."
    }
    return details
    
# Resize image function
def resize_image(image_path, width):
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    new_height = int(width * aspect_ratio)
    img = img.resize((width, new_height))
    return img

# Navigation buttons with custom HTML/CSS for a polished look
col1, _, col2 = st.columns([1,2.6, 1], gap="large")

button_style = """
<style>
.button {
    background-color: #007BFF;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    border-radius: 5px;
    cursor: pointer;
    text-align: center;
    width: 100%;
    margin-top: 10px;
}
.button:hover {
    background-color: #0056b3;
}
</style>
"""

# Inject custom styles
st.markdown(button_style, unsafe_allow_html=True)

with col1:
    if st.button("⬅️ Previous", key="prev_button"):
        if st.session_state.current_step > 0:
            st.session_state.current_step -= 1

with col2:
    if st.button("Next ➡️", key="next_button"):
        if st.session_state.current_step < len(steps) - 1:
            st.session_state.current_step += 1

st.markdown("> 🛠️ **Building a brighter future with AI tools!**")


# Current step content
current = steps[st.session_state.current_step]

# Styled output for the current step
st.markdown(
    f"""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; text-align: center;">
        <h2 style="color: #007BFF;">{current}</h2>
    </div>
    """,
    unsafe_allow_html=True,
)


if current == "Home/Welcome Page":


    st.markdown("""
    ### 🌟 Welcome to Quanti-Test!
    **The Quanti-Test System** is a comprehensive solution designed for precise and efficient allergy testing using skin prick methods.

    It combines **innovative tools** and **workflows** to deliver accurate, user-friendly results, ensuring both patient safety and tester convenience.
    """)
    
    st.video("https://youtu.be/hFNsQOs18Fs")  # Replace with an actual video URL
    st.button("Get Started")

elif current == "Verify Kit Contents":
    st.write("### Checklist of items in the kit:")

    # Dictionary mapping items to their images
    items_with_images = {
        "Quanti-Wells": "quanti_wells.png",
        "Sharptest Applicators": "applicators_sharptest.png",
        "Quicktest Applicators": "applicators_quicktest.png",
        "Droppers": "droppers.png",
    }

    # Loop through items and display each with its resized image
    for item, image_path in items_with_images.items():
        col1, col2 = st.columns([1, 4])  # Adjust column ratio for layout
        with col1:
            st.checkbox(f"{item} present")
        with col2:
            # Set a fixed width (e.g., 100 pixels) for the image
            if item == "Droppers":
                width = 50
            elif item == "Quanti-Wells":
                width = 300
            else:
                width = 100
            st.image(image_path, caption=item, width=width)

    # Upload image for AI verification
    st.write("### AI Verification:")
    uploaded_image = st.file_uploader("Upload a photo of the kit contents", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Kit Photo", use_container_width=True)

        # Mock AI verification
        st.write("AI Verification Result: All items are verified and present in the kit!")



elif current == "Set Up Quanti-Wells":
 
    # Step 1: Taking out Quanti-Wells and placing them into Quanti-Trays
    st.header("Step 1: Prepare Quanti-Wells")
    st.write("Take the Quanti-Wells out of their sterilized package and place them into the Quanti-Trays.")
    
    # Add an image or illustration for this step (replace with actual image path)
    resized_img = resize_image("setup.png", 300)
    st.image(resized_img, caption="Quanti-Wells setup", use_container_width=False)
    # Checklist to confirm step completion
    wells_ready = st.checkbox("I have taken the Quanti-Wells out and placed them into the Quanti-Trays.")

    # Step 2: Labeling allergens and placing stickers on Quanti-Trays
    st.header("Step 2: Label Quanti-Trays")
    st.write("Label allergens' names on the provided stickers and place them on the Quanti-Trays.")
    
    # Text input for allergen names
    allergens = st.text_input("Enter allergen names (comma-separated):", placeholder="e.g., Pollen, Dust, Pet Dander")
    
    if allergens:
        allergen_list = [name.strip() for name in allergens.split(",")]
        st.write("The allergens you entered are:")
        st.write(allergen_list)
    
    # Checkbox to confirm completion
    labels_done = st.checkbox("I have labeled the stickers and placed them on the Quanti-Trays.")

    # Final confirmation
    if wells_ready and labels_done:
        st.success("Great! You have completed the Quanti-Wells setup.")
    else:
        st.warning("Please complete all steps before proceeding.")
        
    uploaded_image = st.file_uploader("Upload an image of the labeled Quanti-Trays for analysis.")
    if uploaded_image:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image for Analysis", use_container_width=True)
        
        # Mock analysis
        st.write("Analyzing allergen stickers placement...")
        result = mock_template_matching(uploaded_image)
        
        # Display results
        st.subheader("Analysis Results")
        st.write(f"Status: **{result['status']}**")
        st.write(f"Suggestion: {result['suggestion']}")

elif current == "Prepare Quanti-Tray":
    
    # Step 1: Hold the left tray and insert it into the right tray
    st.markdown("#### 1. Hold the left tray and insert it into the right tray")
    left_right_tray_image_path = "insert.png"
    st.image(resize_image(left_right_tray_image_path, width=300), caption="Insert Left Tray into Right Tray", use_container_width=False)
    
    # Step 2: Mark testing set sequence on the Quanti-Traybox
    st.markdown("#### 2. Mark testing set sequence on the Quanti-Traybox")
    test_sequence = st.text_input("Enter your testing set sequence (comma-separated, e.g., positive histamine, cat, dog, mouse):")
    
    # Mock AI Sequence Validator
    if test_sequence:
        standard_sequence = ["A", "B", "C", "D"]  # Example standard sequence
        user_sequence = [x.strip() for x in test_sequence.split(",")]
        if user_sequence == standard_sequence:
            st.success("Sequence matches the standard allergen sequence!")
        else:
            st.error("Sequence does not match the standard allergen sequence. Please check and try again.")
    
    # Step 3: Place allergen via dropper into the wells
    st.markdown("#### 3. Place allergen via dropper into the wells")
    dropper_image_path = "allergen.png"
    st.image(resize_image(dropper_image_path, width=300), caption="Use Dropper to Place Allergens in Wells", use_container_width=False)

elif current == "Load Applicators":

    # Step 1: Take out the sterilized Quick-Test applicators and place them into the wells
    st.markdown("#### 1. Take out the sterilized Quick-Test applicators and place them into the wells")
    applicators_image_path = "applicators_quicktest.png"
    st.image(resize_image(applicators_image_path, width=300), caption="Place Applicators into the Wells", use_container_width=False)

    # Step 2: Align the T-mark side with the T-end of applicators
    st.markdown("#### 2. Align the T-mark side with the T-end of applicators")
    alignment_image_path = "applicator.png"
    st.image(resize_image(alignment_image_path, width=300), caption="Align T-mark Side with T-end", use_container_width=False)

    # Mock AI Alignment Checker
    st.markdown("### AI Alignment Checker")
    
    # User input to check alignment (for mock AI)
    alignment_check = st.radio("Check Applicator Alignment", ["Correct", "Incorrect"], index=1)

    if alignment_check == "Correct":
        st.success("The applicators are correctly aligned!")
    else:
        st.error("The applicators are misaligned. Please check the T-mark and T-end alignment.")

        
elif current == "Prepare Skin Test Area":

    # Step 1: Select a flat skin surface
    st.markdown("#### 1. Select a flat skin surface")
    st.write(
        """
        Choose a flat and accessible area for testing, such as:
        - **Volar forearm** (inner forearm)  
        - **Back**  
        Avoid areas with excessive hair growth or uneven surfaces for accurate results.
        """
    )
    #skin_area_image_path = "path_to_skin_surface_image.png"
    #st.image(resize_image(skin_area_image_path, width=300), caption="Recommended Skin Areas", use_container_width=False)

    # Step 2: Apply alcohol or other antiseptics
    st.markdown("#### 2. Sterilize the test area")
    st.write(
        """
        Use alcohol or antiseptics to thoroughly clean the selected test area.
        Wait until the area is completely dry before proceeding to avoid interference.
        """
    )
    #antiseptic_image_path = "path_to_antiseptic_image.png"
    #st.image(resize_image(antiseptic_image_path, width=300), caption="Sterilizing the Test Area", use_container_width=False)

    # Optional: Skin Surface Analysis Mock
    st.markdown("### Skin Surface Analyzer")
    suitability = st.radio(
        "Is the selected skin surface suitable for testing?",
        ["Suitable (flat and clean)", "Unsuitable (hairy or uneven)"],
        index=0
    )

    if suitability == "Suitable (flat and clean)":
        st.success("The selected skin area is suitable for testing!")
    else:
        st.warning("Please select a flat, clean surface and avoid areas with excessive hair.")
        

elif current == "Apply Test":


    # Step 1: Place Quick-Test on the skin
    st.markdown("#### 1. Place Quick-Test on the skin")
    st.write(
        """
        Position the Quick-Test on the selected test area, ensuring it aligns with the prepared skin surface.
        """
    )

    # Step 2: Press the right row gently
    st.markdown("#### 2. Press the right row")
    st.write(
        """
        Apply gentle pressure on the right row for 1-2 seconds. Ensure consistent pressure for accurate results.
        """
    )
    # Display image for this step (replace 'path_to_image' with the actual image path)
    st.image('right.png', caption="Press the right row", width=200)

    # Button to start the timer for the right row
    if st.button("Start Timer for Right Row"):
        st.write("Apply pressure for 2 seconds...")
        time.sleep(2)  # Wait for 2 seconds
        # Play sound notification after 2 seconds
        play_sound_html(base64_sound) 
        st.success("Left row pressure complete!")

    # Step 3: Press the left row gently
    st.markdown("#### 3. Press the left row")
    st.write(
        """
        Apply gentle pressure on the left row for 1-2 seconds.
        """
    )
    # Display image for this step (replace 'path_to_image' with the actual image path)
    st.image('left.png', caption="Press the left row", width=200)

    # Button to start the timer for the left row
    if st.button("Start Timer for Left Row"):
        st.write("Apply pressure for 2 seconds...")
        time.sleep(2)  # Wait for 2 seconds
        # Play sound notification after 2 seconds
        play_sound_html(base64_sound) 
        st.success("Right row pressure complete!")


    # Guidance note
    st.info(
        """
        Make sure to apply consistent pressure and avoid over-pressing to ensure accurate test results.
        """
    )

elif current == "Record and Analyze Results":
    st.write("### Record and Analyze Results")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        # Save the uploaded image locally
        input_image_path = os.path.join("temp", "uploaded_image.png")
        os.makedirs("temp", exist_ok=True)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_image.read())

        st.image(input_image_path, caption="Uploaded Image", use_column_width=True)

        # Segmentation process
        if st.button("Analyze Image"):
            st.write("Processing image...")

            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(
                model_name="UnetPlusPlus",
                encoder_name="resnet34",
                checkpoint_path="segmentation_model_final.pth",
                device=device
            )

            # Define image transform
            transform = transforms.Compose([
                transforms.Resize((736, 736)),
                transforms.ToTensor(),
            ])

            # Perform segmentation
            original_image, segmented_mask = segment_image(model, input_image_path, transform, device)


            # Save segmented output
            output_image_path = os.path.join("temp", "segmented_image.png")
            save_segmented_image(original_image, segmented_mask, output_image_path)

            # Display segmented output
            st.image(output_image_path, caption="Segmented Output", use_column_width=True)

            # Provide download link for segmented image
            with open(output_image_path, "rb") as file:
                st.download_button(
                    label="Download Segmented Image",
                    data=file,
                    file_name="segmented_image.png",
                    mime="image/png"
                )


# elif current == "Record and Analyze Results":
#     # Paths to uploaded files (mocked input and output images, and PDF template)
#     input_image_path = r'C:\Users\Muhammad\Desktop\job\Maverick AI\allergy-test-streamlit\input_image.png'  # Input picture
#     output_image_path = r'C:\Users\Muhammad\Desktop\job\Maverick AI\allergy-test-streamlit\out_image.png' # Output picture
#     pdf_template_path = "results.pdf"  # PDF template
#
#     # Instructions
#     st.write("### 1. Avoid smearing of test allergens to adjacent test sites.")
#     st.write("### 2. Wait 15 minutes.")
#
#     # Progress bar timer
#     progress_bar = st.progress(0)
#     for i in range(101):
#         time.sleep(0.1)  # Mock 10 seconds for simplicity
#         progress_bar.progress(i)
#
#     st.success("Time's up! Analyze the results now.")
#
#     # Positive reaction criteria
#     st.write("### Positive Reaction Criteria")
#     st.write("- Erythema >5mm")
#     st.write("- Wheal >2mm")
#     st.image("positive.png", caption="Example of Positive Reaction", width=300)  # Replace with actual path
#
#     # AI Imaging
#     st.write("### AI Imaging")
#     if st.button("Take Picture and Analyze"):
#         st.image(input_image_path, caption="Input Picture", width=300)
#         st.image(output_image_path, caption="Generated Output Picture", width=300)
#
#
#     # Export Results
#     st.write("### Export Results")
#     if st.button("Export Results as PDF"):
#         with open(pdf_template_path, "rb") as pdf_file:
#             st.download_button("Download PDF", data=pdf_file, file_name="Test_Results.pdf", mime="application/pdf")
            
elif current == "Manage Medication Interference":
        
    st.write("Add medications taken by the patient and a comment about medications that may interfere with test results.")
    
    # Input fields for medications and comment
    medications = st.text_input("Medications Taken by the Patient", help="Enter the medications the patient is currently taking.")

    # Predefined list of medications that may interfere with test results
    interfering_medications = ["antihistamines", "antiemetics", "tranquilizers"]
    
    medication_comment = """
    e.g., antihistamines, antiemetics, tranquilizers
    Provide a comment about medications that could interfere with test results.
    """

    # Check if any of the medications entered by the user matches the interfering list
    if medications:
        # Convert input to lowercase and check for interfering medications
        medications_list = [med.strip().lower() for med in medications.split(",")]
        interfering_found = any(med in interfering_medications for med in medications_list)
        
        if interfering_found:
            st.warning("This medication may interfere with the test results. Please review carefully.")
            
            
elif current == "Results Summary":
                
    # Section for Report Generation
    st.write("### Report Generation")
    st.write("Generate a detailed report of the test results including all relevant data.")
    st.write("You can download the full report as a PDF document or view a summary.")

    # Placeholder for report generation button (mocked functionality)
    if st.button("Generate Report"):
        st.write("Generating the report...")
        # You can add actual report generation logic here or provide a download button if a report is available

    # Section for Actionable Insights
    st.write("### Actionable Insights")
    st.write("Key insights and recommendations based on the test results.")
    st.write("- Based on the allergen reaction, consider further testing or intervention.")
    st.write("- Review medications that may interfere with the results.")

    # Placeholder for actionable insights (mock content)
    actionable_insights = """
    - If a positive reaction occurs, follow up with a specialist.
    - Ensure to check if any medications were taken that might alter the results.
    """
    st.text_area("Actionable Insights", actionable_insights, height=150, disabled=True)

    # Section for Data Trends
    st.write("### Data Trends")
    st.write("Analyze trends based on the test data collected over time.")
    st.write("Here, you can view historical data and see how results have changed.")

    # Placeholder for data trends (mock chart or data visualization)
    # For example, you could plot a simple trend graph (here we use a static mockup for simplicity)
    st.write("Visualizing trends...")

    # Placeholder for a chart or plot (you can use `st.line_chart()` or other visualizations)


    # Generate mock data for trends
    data = pd.DataFrame({
        'Days': np.arange(1, 11),
        'Reaction Level': np.random.randint(1, 10, size=10)
    })

    # Display mock trend chart
    st.line_chart(data.set_index('Days'))            
            


st.sidebar.markdown(
    f"""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
        <h3 style="color: #4CAF50;">Step {st.session_state.current_step + 1}/{len(steps)}</h3>
        <p style="font-size: 16px; font-weight: bold; color: #007BFF;">{current}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

progress = (st.session_state.current_step + 1) / len(steps)
st.sidebar.progress(progress)

# Sidebar for the chatbot UI
st.sidebar.title("Chatbot 🤖")

prompt = st.sidebar.chat_input("Say something")
if prompt:
    st.sidebar.write("Hi! I'm your Quanti-Test Assistant 🤖. How can I help you today?")
