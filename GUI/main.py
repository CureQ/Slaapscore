# Bestandsnaam: main.py 
# Voor het laatst bewerkt op: 02-06-2025

# Importeren van beodigde pakkages
import argparse
import os
from pyshortcuts import make_shortcut
from importlib.metadata import version

# Importeren van een functie uit een ander py bestand
from gui import start_GUI

# Definieert een functie met de naam launch_gui
def launch_gui():
    """GUI launch function"""
    # Roept de functie start_GUI aan
    start_GUI()

# Definieert een functie met de naam "create_shortcut"
def create_shortcut():
    try:
        # Probeer het pad van het huidige pybestand op te halen
        script_path = str(os.path.abspath(__file__))
        # Importeer de functie "make_shortcut" uit de pyshortcuts-bibliotheek
        from pyshortcuts import make_shortcut
        
        # Maakt de app/snelkoppeling aan
        make_shortcut(script=script_path, # Verwijzing naar het script dat moet worden uitgevoerd
                      name="Slaapscore", # Naam van de app
                    #   icon=os.path.join(os.path.dirname(__file__), "MEAlytics_logo.ico"),
                      desktop=True, # Zorgt dat de app/snelkoppeling op het bureaublad wordt geplaatst
                      startmenu=True) # Zorgt dat de app/snelkoppeling ook in het startmenu komt
        
        # Print een melding wanneer de app/snelkoppeling is aangemaakt
        print("Succesfully created desktop shortcut")
    except Exception as error:
        # Print een foutmelding
        print(f"Failed to create shortcut:\n{error}")

def print_version():
    # Print de naam van de app en het versienummer (als er meerdere versienummers zijn)
    print(f"Slaapscore - Version: {version('CureQ')}")

# Hoofdfunctie
def main():
    # Parser wordt aangemaakt, daarmee kunnen extra argumenten worden meegegeven als het programma opstart. 
    parser = argparse.ArgumentParser(description='Launch Slaapscore GUI')
    # Als het programma wordt opgestart met het argument "--create-shortcut" weet het programma dat het een app/snelkoppeling moet maken
    parser.add_argument('--create-shortcut', action='store_true', help='Create a desktop shortcut')
    # parser.add_argument('--version', action='store_true', help='Add shortcut to Start Menu')
    # Check of er een argument is meegegeven
    args = parser.parse_args()
    
    # Als "--create-shortcut" is getypt in de terminal, dan wordt de functie create_shortcut aangeroepen
    if args.create_shortcut:
        create_shortcut()
    # elif args.version:
    #     print_version()
    # Als er geen extra argument is meegegeve, dan wordt de app gestart
    else:
        launch_gui()

# Zorg ervoor dat de code hieronder alleen wordt uitgevoerd als dit bestand direct wordt gerund.
if __name__ == '__main__':
    try:
        # Probeer het programma te starten door de main functie uit te voeren
        main()
    except Exception as error:
        # Print een foutmelding als er iets fout gaat
        print(error)