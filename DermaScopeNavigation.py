import streamlit as st
import time
import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
import math
import uuid

from utils import create_allergen_overlay, generate_pdf, load_model, segment_image, \
save_segmented_image, play_sound_html, get_base64_sound, mock_template_matching, resize_image, paginate_images,\
    save_uploaded_file, capture_from_realsense, capture_from_webcam, save_captured_image, load_images, delete_images

sound_file = "beep.mp3"  
base64_sound = get_base64_sound(sound_file)
UPLOAD_DIR = "Trainings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

# Step titles and content
steps = [
    "Capturing Images", "Allergen Setup", "Home/Welcome Page", "Verify Kit Contents", "Prepare Quanti-Trays and Wells",
    "Patient and Physician Information", "Prepare Applicators and Skin Test Area", "Select Panel and Apply Test", "Record and Analyze Results",
    "Generate Pdf", "Manage Medication Interference", "Results Summary",
]

if "current_step" not in st.session_state:
    st.session_state.current_step = 0
 
if "previous_menu_option" not in st.session_state:
    st.session_state.previous_menu_option = None  # Initialize as None

# Sidebar menu for navigation
menu_option = st.sidebar.radio("Go to", ["Capturing Images", "Settings", "Preparation", "Apply Test"])

# Update current step only when the menu option changes
if menu_option != st.session_state.previous_menu_option:
    if menu_option == "Capturing Images":
        st.session_state.current_step = steps.index("Capturing Images")
    elif menu_option == "Settings":
        st.session_state.current_step = steps.index("Allergen Setup")
    elif menu_option == "Preparation":
        st.session_state.current_step = steps.index("Home/Welcome Page")
    elif menu_option == "Apply Test":
        st.session_state.current_step = steps.index("Patient and Physician Information")

    # Update the previous menu option in session state
    st.session_state.previous_menu_option = menu_option


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


# Current step content
current = steps[st.session_state.current_step]



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
st.progress(progress)

# Sidebar for the chatbot UI
st.sidebar.title("Chatbot 🤖")

prompt = st.sidebar.chat_input("Say something")
if prompt:
    st.sidebar.write("Hi! I'm your Quanti-Test Assistant 🤖. How can I help you today?")


st.markdown("> 🛠️ **Building a brighter future with AI tools!**")


# Styled output for the current step
st.markdown(
    f"""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; text-align: center;">
        <h2 style="color: #007BFF;">{current}</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

if "refresh_flag" not in st.session_state:
    st.session_state["refresh_flag"] = False

if "selected_images" not in st.session_state:
    st.session_state["selected_images"] = set()

selected_images = st.session_state["selected_images"]

# Default allergens for each panel
DEFAULT_ALLERGENS = {
        "Panel A": ["Positive histamine", "Cat", "Dog", "Mouse", "Horse", "Cockroach", "Dust Mite Mix", "Penicill Chrys Mold", "Rhizopus Nigra Mold", "Negative Control"],
        "Panel B": ["Cladospor Sphaer Mold", "Alternaria Mold", "Mucor Mix Mold", "Cladospor Herbarum Mold", "Bipolans Mold", "Fusarium solani", "Sweet Gum Tree", "Sycamore East Tree", "Eastern Oak Tree", "Ash Red/Green Tree"],
        "Panel C": ["Birch River Tree", "Cedar Red Tree", "Cotton Wood East Tree", "Elm Amer Tree", "Hickory White Tree", "Maple Red Tree", "Mulberry Red Tree", "Pine White Tree", "Pigweed Rough", "Dock Sorrel Weed"],
        "Panel D": ["English Plantain Weed", "Ragweed Mix", "Baccharis Weed", "Cocklebur Weed", "Lambs Quarter Weed", "Mugwort Common Weed", "Nettle Weed", "Bermuda Grass", "KORT w/SV Grass Mix", "Johnson Grass"],
    }

# Initialize session state with default values if not present
if "allergen_setup" not in st.session_state:
    st.session_state.allergen_setup = DEFAULT_ALLERGENS.copy()

if current == "Capturing Images":
    
    st.write("")
    st.write("")
    
    st.header("Upload Images")
    uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        uploaded_names = []
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(UPLOAD_DIR, uploaded_file)
            uploaded_names.append(uploaded_file.name)

        if uploaded_names:
            st.success(f"Uploaded: {', '.join(uploaded_names)}")

    # Camera Capture Section
    st.header("Capture Image from Intel RealSense Camera")
    if st.button("Capture from Intel RealSense Camera"):
        captured_image = capture_from_realsense(st)
        if captured_image:
            file_name  = f"imageD405_{uuid.uuid4().hex}.png"
            save_captured_image(captured_image, UPLOAD_DIR, file_name)
            st.image(captured_image, caption="Captured Image from RealSense", use_container_width=True)
            st.success(f"Captured image saved as {file_name}")

    if st.button("Capture from Webcam"):
        captured_image = capture_from_webcam(st)
        if captured_image:
            file_name = f"image_{uuid.uuid4().hex}.png"
            save_captured_image(captured_image, UPLOAD_DIR, file_name)
            st.image(captured_image, caption="Captured Image from Webcam", use_container_width=True)
            st.success(f"Captured image saved as {file_name}")

    # Display Images
    st.header("Display Images")
    
    if st.button("Delete Selected Images") and selected_images:
        delete_images(UPLOAD_DIR, selected_images)
        st.success("Deleted selected images.")
        st.session_state["selected_images"] = set()  # Clear the selection
        st.session_state["refresh_flag"] = not st.session_state["refresh_flag"]  # Toggle the flag


    images = load_images(UPLOAD_DIR)
    
    if images:
        
        if "selected_images" not in st.session_state:
            st.session_state["selected_images"] = set()

        selected_images = st.session_state["selected_images"]

        # Pagination
        per_page = 10
        total_pages = math.ceil(len(images) / per_page)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, label_visibility="hidden")
        paginated_images = paginate_images(images, page, per_page)

        # Display images (one image per line with uniform size)
        for idx, image_name in enumerate(paginated_images):
            image_path = os.path.join(UPLOAD_DIR, image_name)
            resized_image = resize_image(image_path, width=100, height=100)  # Resize to 100x100
            col1, col2 = st.columns([1, 9])  # Create two columns for layout
            with col1:
                if st.checkbox(f"{idx + 1}", key=f"chk_{image_name}", value=image_name in selected_images):
                    selected_images.add(image_name)
                else:
                    selected_images.discard(image_name)
            with col2:
                st.image(resized_image, use_container_width=False)
                st.write(image_name)  # Display the image name horizontally
       
elif current == "Allergen Setup":

    st.write("")
    st.write("")

    # Dropdown to select a panel
    selected_panel = st.selectbox(
        "Select a panel to set up:",
        options=['Panel A', 'Panel B', 'Panel C', 'Panel D'],
        key="selected_panel_dropdown"
    )

    # Extract the selected panel letter (A, B, C, or D)
    panel_letter = selected_panel.split()[-1]
    panel_key = f"Panel {panel_letter}"

    # Get current allergens for the selected panel and initialize with default values
    temp_allergens = st.session_state.allergen_setup[panel_key].copy()  # Deep copy

    # Create two columns
    cols = st.columns([1, 1])  # Adjust the width ratio of the columns

    # Left column: Allergen input fields
    with cols[0]:
        st.write("### Allergen Input Fields")
        for i in range(10):
            allergen_key = f"{panel_letter}_allergen_{i}"
            temp_allergens[i] = st.text_input(
                f"Allergen {i + 1}",
                key=allergen_key,
                value=temp_allergens[i]
            )

    # Right column: Allergen mapping image
    with cols[1]:
        st.write("### Allergen Mapping")
        applicator_image_path = "applicatortop.png"  # Path to your image
        if temp_allergens:
            overlay_image = create_allergen_overlay(applicator_image_path, temp_allergens)
            st.image(overlay_image, caption=f"{panel_key} Allergen Mapping")
        else:
            st.error(f"No allergens found for {panel_key}. Please set them up.")

    # Save button for allergen setup
    if st.button("Save Allergen Setup"):
        st.session_state.allergen_setup[panel_key] = temp_allergens  # Ensure the updated list is saved
        st.success(f"Allergen setup for {panel_key} saved successfully!")


elif current == "Home/Welcome Page":

    st.markdown("""
    ### 🌟 Welcome to Quanti-Test!
    **The Quanti-Test System** is a comprehensive solution designed for precise and efficient allergy testing using skin prick methods.

    It combines **innovative tools** and **workflows** to deliver accurate, user-friendly results, ensuring both patient safety and tester convenience.
    """)
    
    st.video("https://youtu.be/hFNsQOs18Fs")  # Replace with an actual video URL

elif current == "Verify Kit Contents":
    
    st.write("")
    st.write("")
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
    #st.write("### AI Verification:")
    #uploaded_image = st.file_uploader("Upload a photo of the kit contents", type=["png", "jpg", "jpeg"])
    #if uploaded_image:
        # Display uploaded image
    #    st.image(uploaded_image, caption="Uploaded Kit Photo", use_container_width=True)

        # Mock AI verification
     #   st.write("AI Verification Result: All items are verified and present in the kit!")


elif current == "Prepare Quanti-Trays and Wells":

    st.write("")
    st.write("")
    
    # Substep 1: Prepare Quanti-Wells
    st.markdown("#### 1. Prepare Quanti-Wells")
    st.write("Take the Quanti-Wells out of their sterilized package and place them into the Quanti-Trays.")
    
    # Add image for this substep
    quanti_wells_image_path = "setup.png"
    st.image(resize_image(quanti_wells_image_path, width=300), caption="Quanti-Wells Setup", use_container_width=False)
    
    # Checkbox to confirm completion
    wells_ready = st.checkbox("I have taken the Quanti-Wells out and placed them into the Quanti-Trays.")
    
    # Substep 2: Label Trays with Panels
    st.markdown("#### 2. Label Quanti-Trays")
    st.write("Label the Quanti-Trays with the corresponding panel (e.g., Panel A, Panel B, Panel C, or Panel D).")
    
    # Checkbox to confirm labeling is done
    labels_done = st.checkbox("I have labeled the Quanti-Trays with the selected panel.")

    st.markdown("#### 3. Hold the Left Tray and Insert It into the Right Tray")
    left_right_tray_image_path = "insert.png"
    st.image(resize_image(left_right_tray_image_path, width=300), caption="Insert Left Tray into Right Tray", use_container_width=False)

    insert_done = st.checkbox("I have inserted Left Tray into Right Tray.")
    
    # Substep 2: Place allergen via dropper into the wells
    st.markdown("#### 4. Place Allergen via Dropper into the Wells")
    dropper_image_path = "allergen.png"
    st.image(resize_image(dropper_image_path, width=300), caption="Use Dropper to Place Allergens in Wells", use_container_width=False)

    drop_done = st.checkbox("I have used droppers to place allergens in wells")
    
    # Final Confirmation
    st.markdown("Final Confirmation")
    if wells_ready and labels_done and insert_done and drop_done:
        st.success("All steps for setting up and preparing the Quanti-Wells and Quanti-Trays are complete!")
    else:
        st.warning("Please complete all steps before proceeding.")

  
elif current == "Patient and Physician Information":

    # Initialize session state variables if not already present
    if "sheet_title" not in st.session_state:
        st.session_state.sheet_title = "Allergy Testing Sheet"
    if "product_name" not in st.session_state:
        st.session_state.product_name = "Quanti-Test"
    if "patient_info" not in st.session_state:
        st.session_state.patient_info = {
            "name": "",
            "id": "",
            "dob": None,
            "location": "Back",
        }
    if "physician_info" not in st.session_state:
        st.session_state.physician_info = {
            "name": "",
            "address": "",
            "telephone": "",
            "email": "",
        }

    st.write("")
    st.write("")
    st.header("Titles")
    st.text_input("Allergy Testing Sheet Title", value=st.session_state.sheet_title)
    st.text_input("Product Name", value=st.session_state.product_name)

    st.header("Patient Information")
    st.session_state.patient_info["name"] = st.text_input(
        "Patient Name", key="patient_name", value=st.session_state.patient_info["name"]
    )
    st.session_state.patient_info["id"] = st.text_input(
        "Patient ID", key="patient_id", value=st.session_state.patient_info["id"]
    )
    st.session_state.patient_info["dob"] = st.date_input(
        "Date of Birth", key="patient_dob", value=st.session_state.patient_info["dob"]
    )
    st.session_state.patient_info["location"] = st.selectbox(
        "Test Location", ["Back", "Arm"], key="patient_location", 
        index=["Back", "Arm"].index(st.session_state.patient_info["location"])
    )

    st.header("Physician Information")
    st.session_state.physician_info["name"] = st.text_input(
        "Physician Name", key="physician_name", value=st.session_state.physician_info["name"]
    )
    st.session_state.physician_info["address"] = st.text_input(
        "Address", key="physician_address", value=st.session_state.physician_info["address"]
    )
    st.session_state.physician_info["telephone"] = st.text_input(
        "Telephone", key="physician_telephone", value=st.session_state.physician_info["telephone"]
    )
    st.session_state.physician_info["email"] = st.text_input(
        "Email", key="physician_email", value=st.session_state.physician_info["email"]
    )

    # Save button
    if st.button("Save Information"):

        st.session_state.patient_info.update({
            "name": st.session_state.patient_info["name"],
            "id": st.session_state.patient_info["id"],
            "dob": st.session_state.patient_info["dob"],
            "location": st.session_state.patient_info["location"]
        })

        st.session_state.physician_info.update({
            "name": st.session_state.physician_info["name"],
            "address": st.session_state.physician_info["address"],
            "telephone": st.session_state.physician_info["telephone"],
            "email": st.session_state.physician_info["email"]
        })

        st.success("Information saved successfully!")

elif current == "Prepare Applicators and Skin Test Area":

    st.write("")
    st.write("")

    # Step 1: Load Applicators
    st.markdown("#### 1. Load and Align Applicators")
    st.write(
        """
        - Take out the sterilized Quick-Test applicators from their package.  
        - Place them into the wells and align the **T-mark side** with the **T-end** of the applicators.
        """
    )
    # Display images for applicator setup and alignment
    applicators_image_path = "applicators_quicktest.png"
    alignment_image_path = "applicator.png"
    st.image(resize_image(applicators_image_path, width=300), caption="Place Applicators into the Wells", use_container_width=False)
    st.image(resize_image(alignment_image_path, width=300), caption="Align T-mark Side with T-end", use_container_width=False)

    # Mock AI Alignment Checker
    st.markdown("### Applicator Alignment Checker")
    alignment_check = st.radio("Check Applicator Alignment", ["Correct", "Incorrect"], index=1)
    if alignment_check == "Correct":
        st.success("The applicators are correctly aligned!")
    else:
        st.error("The applicators are misaligned. Please check the T-mark and T-end alignment.")

    # Step 2: Prepare Skin Test Area
    st.markdown("#### 2. Prepare the Skin Test Area")
    st.write(
        """
        - Choose a flat and accessible area for testing, such as:
          - **Volar forearm** (inner forearm)  
          - **Back**  
        - Avoid areas with excessive hair growth or uneven surfaces for accurate results.
        - Sterilize the selected area using alcohol or antiseptics.  
        - Ensure the area is completely dry before proceeding.
        """
    )

    # Optional: Skin Surface Analysis Mock
    st.markdown("### Skin Surface Analyzer")
    suitability = st.radio(
        "Is the selected skin surface suitable for testing?",
        ["Suitable (flat and clean)", "Unsuitable (hairy or uneven)"],
        index=1
    )
    if suitability == "Suitable (flat and clean)":
        st.success("The selected skin area is suitable for testing!")
    else:
        st.warning("Please select a flat, clean surface and avoid areas with excessive hair.")

    # Final Confirmation
    if alignment_check == "Correct" and suitability == "Suitable (flat and clean)":
        st.success("The applicators are loaded, aligned, and the skin test area is ready!")
    else:
        st.warning("Please complete all steps correctly before proceeding.")
    

elif current == "Select Panel and Apply Test":

    # Path to the applicator image
    applicator_image_path = "applicatortop.png"

    # List of all panels
    panels = ['A', 'B', 'C', 'D']
    panel_images = {}

    # Generate panel images with allergen overlays
    for panel in panels:
        panel_key = f"Panel {panel}"
        allergens = st.session_state.allergen_setup.get(panel_key, [])
        if allergens:
            overlay_image = create_allergen_overlay(applicator_image_path, allergens)
            panel_images[panel] = overlay_image
        else:
            panel_images[panel] = None

    st.write("")
    st.write("")
    # Dropdown to select panel
    selected_panel = st.selectbox(
        "Select a panel to proceed:",
        options=[f"Panel {panel}" for panel in panels],
        key="panel_selection"
    )

    # Extract selected panel details
    panel_letter = selected_panel.split()[-1]  # Extract panel letter (A, B, C, or D)
    allergens = st.session_state.allergen_setup.get(f"Panel {panel_letter}", [])
    selected_image = panel_images.get(panel_letter)

    # Display panel image and allergen list
    col1, col2 = st.columns([1, 1])  # Adjust column width
    with col2:
        if selected_image:
            st.image(selected_image, caption=f"{selected_panel} Allergen Mapping", use_container_width=True)
        else:
            st.error(f"No allergens found for {selected_panel}. Please set them up in 'Settings'.")

    with col1:
        if allergens:
            st.markdown(f"### Allergens in {selected_panel}")
            for idx, allergen in enumerate(allergens, start=1):
                st.write(f"{idx}. {allergen}")
        else:
            st.warning("No allergens defined for this panel.")

    # Save the selected panel
    if st.button("Save Selected Panel"):
        st.session_state.selected_panel = selected_panel
        st.session_state.selected_allergens = allergens
        st.session_state.selected_panel_image = selected_image
        st.success(f"Saved {selected_panel} with allergens!")
    

    st.divider()

    # Apply Test Steps
    st.subheader("Apply Test")

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
    # Display image for this step
    st.image('right.png', caption="Press the right row", width=200)

    # Step 3: Press the left row gently
    st.markdown("#### 3. Press the left row")
    st.write(
        """
        Apply gentle pressure on the left row for 1-2 seconds.
        """
    )
    # Display image for this step
    st.image('left.png', caption="Press the left row", width=200)


    # Step 4: Wait for 15 minutes
    st.markdown("#### 4. Wait for 15 minutes")
    st.write(
        """
        After applying the test, wait for 15 minutes to allow the allergens to interact with the skin.
        Use this time to prepare the camera for taking a photo of the test area.
        """
    )

    if st.button("Start Test Timing"):
            # Step 1: Press the right row
            st.write("Press the right row for 2 seconds...")
            play_sound_html(base64_sound, st)  # Play beep
            time.sleep(2)

            # Step 2: Press the left row
            st.write("Press the left row for 2 seconds...")
            play_sound_html(base64_sound, st)  # Play beep
            time.sleep(2)

            # Step 3: Wait for 15 minutes
            st.write("Waiting for 15 minutes. Timer started...")
            progress_text = st.empty()
            progress_bar = st.progress(0)

            for i in range(60, 0, -1):  # 900 seconds = 15 minutes
                minutes, seconds = divmod(i, 60)
                progress_text.markdown(f"**Time remaining: {minutes} minutes {seconds} seconds**")
                progress_bar.progress((60 - i) / 60)
                time.sleep(1)

            # Notify when 15 minutes are up
            progress_text.markdown("**Time's up! Please take a photo of the test area.**")
            progress_bar.progress(1)
            play_sound_html(base64_sound, st)  # Play final beep

    # Guidance note
    st.info(
        """
        Ensure you take a clear and focused photo of the test area for accurate analysis. Avoid disturbing the test during this waiting period.
        """
    )


elif current == "Record and Analyze Results":

    # Retrieve the selected panel and its allergens
    selected_panel = st.session_state.get("selected_panel", "")
    panel_letter = selected_panel.replace("Panel ", "")
    selected_image = st.session_state.get("selected_panel_image")  # Retrieve the selected image
    allergens = st.session_state.get("selected_allergens")
    
    st.write("")
    st.write("")
    # Display the selected panel's image and allergens
    col1, col2 = st.columns([1, 1])  # Adjust column width
    with col2:
        if selected_image:
            st.image(selected_image, caption=f"{selected_panel} Allergen Mapping", use_container_width=True)
        else:
            st.error(f"No selected panel image found. Please ensure you have completed the 'Select Panel and Apply Test' step.")

    with col1:
        if allergens:
            st.markdown(f"### Allergens in {selected_panel}")
            for idx, allergen in enumerate(allergens, start=1):
                st.write(f"{idx}. {allergen}")
        else:
            st.warning("No allergens defined for this panel.")
            
    # Upload image
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        # Save the uploaded image locally
        input_image_path = os.path.join("temp", "uploaded_image.png")
        os.makedirs("temp", exist_ok=True)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_image.read())

        st.image(input_image_path, caption="Uploaded Image", use_container_width=True)

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
            st.image(output_image_path, caption="Segmented Output", use_container_width=True)

            # Provide download link for segmented image
            with open(output_image_path, "rb") as file:
                st.download_button(
                    label="Download Segmented Image",
                    data=file,
                    file_name="segmented_image.png",
                    mime="image/png"
                )
                
            # Record and analyze results for the selected panel
            if "results" not in st.session_state:
                st.session_state.results = {}

            selected_panel = st.session_state.get("selected_panel", "").replace("Panel ", "")
            panel_key = f"Panel {selected_panel}"  # Correctly format the key
            st.subheader(panel_key)

            if panel_key in st.session_state.allergen_setup:
                sites = []
                for i, allergen in enumerate(st.session_state.allergen_setup[panel_key], start=1):
                    well = st.number_input(f"{allergen} - Well (mm)", key=f"{selected_panel}_{i}_well", step=0.1)
                    sites.append({"index": i, "allergen": allergen, "well": well})

                # Append new data to results for the selected panel
                if panel_key in st.session_state.results:
                    st.session_state.results[panel_key].extend(sites)
                else:
                    st.session_state.results[panel_key] = sites

                st.success(f"Results for {panel_key} updated successfully!")
            else:
                st.error(f"No allergens found for {panel_key}. Please ensure the panel setup is complete.")


elif current == "Generate Pdf":
    
    if "results" not in st.session_state:
        st.session_state.results = {
            "Panel A": [],
            "Panel B": [],
            "Panel C": [],
            "Panel D": [],
        }
        
    pdf_buffer = generate_pdf(
        st.session_state.sheet_title,
        st.session_state.product_name,
        st.session_state.patient_info,
        st.session_state.physician_info,
        st.session_state.results,
    )
    st.download_button("Download Allergy Test PDF", pdf_buffer, "allergy_test.pdf", "application/pdf")
       
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
            

