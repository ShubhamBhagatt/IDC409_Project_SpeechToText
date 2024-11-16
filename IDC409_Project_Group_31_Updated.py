# Importing the required libraries for speech recognition and audio processing
# Required Python libraries to run this program:
# Install them by using the following commands if they are not already installed:

# pip install SpeechRecognition
# pip install PyAudio
# install pyttsx3

import speech_recognition as sr
import pyaudio
from datetime import datetime  # Added this import for timestamp

# Default language setting (Indian English)
DEFAULT_LANGUAGE = "en-IN"

# Function to convert speech to text using the microphone in real-time
def speech_to_text():
    r = sr.Recognizer()  # Initialize recognizer
    while True:
        try:
            with sr.Microphone() as source2:
                print("Adjusting for ambient noise...")  # Adjust for ambient noise in the environment
                r.adjust_for_ambient_noise(source2, duration=1)

                print("Listening... (Command 'terminate program' to stop the conversation)")
                audio2 = r.listen(source2)  # Listen for audio input

                # Convert audio to text using Google's speech recognition
                MyText = r.recognize_google(audio2, language=DEFAULT_LANGUAGE)
                MyText = MyText.lower()  # Convert text to lowercase for consistency

                print("----> You said: ----> ", MyText)

                # Added: Save the recognized text to a file
                with open("speech_record.txt", "a") as file:
                    file.write(f"{datetime.now()}: {MyText}\n")

                # Check if user wants to terminate the program by saying "terminate program"
                if 'terminate program' in MyText:
                    print("Terminating program...")
                    break

        except sr.RequestError as e:
            # Handle exceptions related to API request failures
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            # Handle cases where the speech could not be understood
            print("Sorry, I could not understand your speech. Please try again.")

# Function to convert audio file to text (default language used)
def audio_to_text(file):
    all_ = []  # Initialize an empty list to store recognized text
    r = sr.Recognizer()  # Create a Recognizer instance
    try:
        # Remove extra quotes around the file path if present
        file = file.strip('"')
        with sr.AudioFile(file) as source:
            # Load the entire audio file into memory
            audio_data = r.record(source)  # Record the audio data from the file
            print("Converting audio to text...")

            # Try to recognize the spoken text from the audio file
            try:
                text = r.recognize_google(audio_data, language=DEFAULT_LANGUAGE)
                all_.append(text)  # Append recognized text to the list
                print("Lyrics/Spoken Text from Audio: ", text)
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    except Exception as e:
        # Handle general exceptions, e.g., if file is not found
        print(f"Error processing file: {e}")

    return all_

# Main function to offer a menu-based interface to the user
def main():
    while True:
        print("Choose an option:")
        print("1. Run Speech-to-Text (Microphone Input)")
        print("2. Run Audio-to-Text (File Input)")

        choice = input("Enter 1 or 2 (or '0' to quit): ")

        # Perform the selected operation based on user input
        if choice == '1':
            speech_to_text()
        elif choice == '2':
            while True:
                file_path = input("Enter the file path of the audio file (or '0' to quit): ")
                if file_path == '0':
                    print("Terminating program...")
                    break
                audio_to_text(file_path)
        elif choice == '0':
            print("Terminating program...")
            break
        else:
            print("Invalid choice. Please enter 1 or 2, or '0' to quit.")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()