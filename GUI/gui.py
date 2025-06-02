# Bestandsnaam: gui.py
# Geschreven door: Esmee Springer
# Voor het laatst bewerkt op: 26-05-2025

# Importeren van de benodigde pakkages
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
from tabulate import tabulate
import tkinter as tk
from analyse_ECG_files_beide_gui import run_ecg_analysis
from Overzicht_movesense import run_movesense_analysis

# Instellen van de (achtergrond) kleur van het systeem en van de knoppen
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")

# Initialiseer hoofdvenster van de applicatie met titel en vaste afmeting
class App(ctk.CTk):
    def __init__(self):
        # Initialiseer het hoofdvenster via de superklasse
        super().__init__()
        # Geef een titel aan het hoofdvenster
        self.title("MoveSense Slaap Analyse")
        # Stel de grootte van het venster in op 1000x800 pixels
        self.geometry("1000x800")

        # Maak een tabbladstructuur aan met behulp van ttk.Notebook
        self.notebook = ttk.Notebook(self)
        # Zorg dat de tabbladen het hele venster opvullen
        self.notebook.pack(fill="both", expand=True)

        # Eerste tabblad voor de analyse van de MoveSense data en het generenen van het totaaloverzicht wordt aangemaakt
        self.tab1 = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.tab1, text="Analyse en totaaloverzicht")

        # Informatieve uitleg over de functionaliteit van beide tabbladen, deze wordt zichtbaar in de app
        self.label_info = ctk.CTkLabel(
            self.tab1,
            text=(
                "Gebruik deze tool om MoveSense-data te analyseren, overzichtsrapporten te genereren en de slaapscore matrix te bekijken.\n"
                "Het venster bestaat uit 2 tabjes linksboven. Het eerste tabje kan gebruikt worden om de MoveSense data te analyseren en om een totaaloverzicht van de resultaten te genereren.\n" \
                "Het tweede tabje kan gebruikt worden om de slaapscore matrix te visualiseren, waarbij de slaapscores (0-100) per participant per nacht worden getoond. \n"
                "Belangrijk: het totaaloverzicht en de slaapscore matrix kunnen pas gegenereerd/getoond worden nadat er tenminste in ieder geval 1 analyse met de MoveSense bestanden is uitgevoerd."
                
            ),
            wraplength=800,
            justify="left"
        )
        # Plaats het informatielabel met ruimte erboven en eronder
        self.label_info.pack(pady=(20, 10))

        # Uitleg en de knop om de map met ruwe MoveSense data te selecteren en de analyse te starten
        self.label_expl1 = ctk.CTkLabel(self.tab1, text="Selecteer hieronder de map met ruwe MoveSense data per participant.\n"
                                        "Ter voorbeeld: Folders met de volgende naam 'Movesense_participant_XX', waarbij XX staat voor het participantnummer.\n"
                                        "\n"
                                        "De resultaten van de gerunde Movesense bestanden worden zowel opgeslagen in de map 'MoveSense_data_resultaten' als in de map 'Movesense_participant_XX'. \n"
                                        "De map 'MoveSense_data_resultaten' wordt gebruikt voor het genereren van het totaaloverzicht. \n"
                                        "Wanneer je een map (Movesense_participant_XX) al hebt gerund hoef je hem niet nog eens te runnen, de resultaten blijven bewaard.\n"
                                        "Wanneer alle Movesense_participant_XX folders zijn gerund, kun je gelijk doorgaan naar het maken van het totaaloverzicht.")
        # Plaats de uitleg onder het informatielabel
        self.label_expl1.pack(pady=(0, 10))
        
        # Button waarmee de gebruiker een map selecteert en de analyse start
        self.btn_analyse = ctk.CTkButton(self.tab1, text="Map selecteren en analyse starten", command=self.start_analysis)
        self.btn_analyse.pack(pady=10)

        # Label dat uitlegt hoe je de map met de eerder gegenereerde Excel-resultaten selecteert
        self.label_expl2 = ctk.CTkLabel(self.tab1, text="Selecteer de map 'MoveSense_data_resultaten' met de eerder gegenereerde Excel-resultaten.")
        self.label_expl2.pack(pady=(10, 10))
        
        # Button om het totaaloverzicht te genereren op basis van de geselecteerde resultaten
        self.btn_overview = ctk.CTkButton(self.tab1, text="Genereer totaaloverzicht", command=self.generate_overview)
        self.btn_overview.pack(pady=10)

        # Label met instructies voor het dropdown-menu op een participant te selecteren en gewenste kolommen te selecteren
        self.dropdown_label = ctk.CTkLabel(self.tab1, text="Selecteer met het dropdown-menu hieronder een participant om het resultaat te bekijken:\n"
                                            "De checkbox aan de linkerkant kan gebruikt worden om de gewenste kolommen te selecteren en zichtbaar te maken.")
        self.dropdown_label.pack(pady=(10, 0))
        
        # Dropdown-menu op deelnemers te selecteren, de resultaten worden getoond wanneer je een deelnemer selecteert
        self.participant_dropdown = ctk.CTkComboBox(self.tab1, values=[], command=self.show_participant_data)
        self.participant_dropdown.pack(pady=5)

        # Hoofdframe waarin kolomkeuze en de tekst naast elkaar worden getoond
        main_frame = ctk.CTkFrame(self.tab1)
        main_frame.pack(pady=10, fill="both", expand=True)

        # Scrollbaar frame voor de checkbox om te gewenste kolommen te selecteren
        self.columns_frame = ctk.CTkScrollableFrame(main_frame, width=220, height=300)
        self.columns_frame.pack(side="left", padx=10, pady=10, fill="y")
        
        # Label boven de checkbox waar je de kolommen kunt selecteren
        self.columns_label = ctk.CTkLabel(self.columns_frame, text="Selecteer gewenste kolommen:")
        self.columns_label.pack(pady=(0, 10))

        # Dictionary voor de checkbox, zorgt voor de dynamische aanpssing wanneer je kolommen (de)selecteert
        self.column_checkboxes = {}

        # Frame voor de tekstbox waar de geselecteerde data in wordt getoond
        text_frame = ctk.CTkFrame(main_frame)
        text_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)
        
        # Scrollbars voor de tekstboxen (voor de tabelweergave), zowel horizontaal als verticaal
        scroll_y = tk.Scrollbar(text_frame, orient="vertical")
        scroll_x = tk.Scrollbar(text_frame, orient="horizontal")
        
        # Tekstvak om data weer te geven met specifiek lettertype
        self.textbox = ctk.CTkTextbox(text_frame, height=300, width=650, wrap="none", corner_radius=8)
        self.textbox.configure(font=("Courier New", 12))
        self.textbox.pack(side="left", fill="both", expand=True)
        
        # Koppel de scrollbars aan de tekstbox om scrollen mogelijk te maken
        scroll_y.config(command=self.textbox.yview)
        scroll_y.pack(side="right", fill="y")
        scroll_x.config(command=self.textbox.xview)
        scroll_x.pack(side="bottom", fill="x")
        
        # Koppel de scrollbalk aan de tekstbox
        self.textbox.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        # Label om statusinformatie onderaan tabblad 1 te tonen
        self.status_label = ctk.CTkLabel(self.tab1, text="", anchor="w", fg_color="transparent")
        self.status_label.pack(fill="x", padx=10, pady=(5, 10))
        self.excel_data = None

        # Maak het tweede tabblad aan voor de slaapscore matrix
        self.tab2 = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.tab2, text="Slaapscore matrix")

        # Label met uitleg over welk bestand geslecteert moet worden en hoe nachten worden geteld
        self.matrix_label = ctk.CTkLabel(self.tab2, text="Selecteer het bestand 'Slaapscore_matrix.xlsx' voor het overzicht van de slaapscore per proefpersoon per nacht.\n"
                                         "\n"
                                         "Wanneer een participant na 00:00 's nachts is gestart met meten, wordt het gezien als de nacht van de dag ervoor.\n"
                                         "Ter voorbeeld: 14/07 01:00 en 14/07 10:00 zijn 2 verschillende nachten, maar hebben dezelfde datum. 14/07 01:00 wordt gerekend tot de nacht van 13/07.\n"
                                         "Dit is gedaan zodat de temperatuur van de nacht vergeleken wordt met de juiste nacht. \n"
                                         "\n"
                                         "Gebruik het dropdown-menu onderaan het venster om een specifieke nacht te selecteren en te bekijken.\n" 
                                         "\n"
                                         "De totale slaapscore matrix wordt opgeslagen in de map 'MoveSense_data' als Excel-bestand genaamd 'Slaapscore_matrix.xlsx'.")
        self.matrix_label.pack(pady=10)

        # frame voor de matrix tekstvak en scrollbars
        matrix_frame = ctk.CTkFrame(self.tab2)
        matrix_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tekstvak waar de data van de matrix in wordt getoond
        self.matrix_textbox = ctk.CTkTextbox(matrix_frame, wrap="none", font=("Courier New", 12))
        self.matrix_textbox.pack(side="left", fill="both", expand=True)
        
        # Verticale scrollbar aan de rechterkant van het tekstvak
        matrix_scroll_y = tk.Scrollbar(matrix_frame, orient="vertical", command=self.matrix_textbox.yview)
        matrix_scroll_y.pack(side="right", fill="y")
        
        # Horizontale scrollbar onderaan het tekstvak
        matrix_scroll_x = tk.Scrollbar(self.tab2, orient="horizontal", command=self.matrix_textbox.xview)
        matrix_scroll_x.pack(side="bottom", fill="x")
        
        # Zorg dat het tekstvak weet dat de scrollbars gekoppeld zijn
        self.matrix_textbox.configure(yscrollcommand=matrix_scroll_y.set, xscrollcommand=matrix_scroll_x.set)

        # Knop om het Slaapscore_matrix.xlsx bestand te laden
        self.btn_matrix = ctk.CTkButton(self.tab2, text="Laad slaapscore matrix", command=self.load_sleep_matrix)
        self.btn_matrix.pack(pady=10)

        # Label voor het dropdown-menu om een datum te selecteren
        self.date_dropdown_label = ctk.CTkLabel(self.tab2, text="Selecteer een datum om scores van die specifieke nacht te bekijken:")
        self.date_dropdown_label.pack(pady=(10, 0))

        # Dropdown-menu met de verschillende nachten om te selecteren
        self.date_dropdown = ctk.CTkComboBox(self.tab2, values=[], command=self.show_scores_by_date)
        self.date_dropdown.pack(pady=(0, 10))

    def start_analysis(self):
        # Open een dialoogvenster om een map te selecteren met MoveSense data
        folder = filedialog.askdirectory(title="Selecteer map met MoveSense data")
        
        # Als er geen map is geselecteerd, stop dan met de functie
        if not folder:
            return
        try:
            # Voer de ECG-analyse uit op de geselecteerde map
            run_ecg_analysis(folder)

            # Informeer de gebruiker dat de analyse klaar is
            messagebox.showinfo("Klaar", "Slaapstadia voorspellingen voltooid en opgeslagen!")
            
            # Update de statuslabel in de GUI om succes aan te geven
            self.status_label.configure(text="Slaapstadia voorspellingen succesvol voltooid.")
        except Exception as e:
            # Laat een foutmelding zien als er iets misgaat tijdens de analyse
            messagebox.showerror("Fout", f"Er is iets misgegaan:\n{e}")
            
            # Update de statuslabel onderaan in de GUI om de fout aan te geven
            self.status_label.configure(text="Fout tijdens slaapstadia voorspellingen.")

    def generate_overview(self):
        # Open een dialoog om de map te selecteren met eerder gegenereerde Excel resultaten
        folder = filedialog.askdirectory(title="Selecteer map met Excel-resultaten")
        
        # Stop wanneer er geen map is geselecteerd
        if not folder:
            return
        try:
            # Voer de MoveSense analyse uit op de geselecteerde map
            run_movesense_analysis(folder)
            
            # Bepaal de bovenliggende map ten opzichte van de geselecteerde map
            bovenliggende_map = os.path.dirname(folder)
            
            # Definieer het pad naar het overzichtsbestand met gefilterde resultaten
            pad = os.path.join(bovenliggende_map, "Overzicht_resultaten_gefilterd.xlsx")
            
            # Controleer of het overzichtsbestand bestaat. Zo niet, geef een foutmelding
            if not os.path.exists(pad):
                messagebox.showerror("Fout", f"Bestand niet gevonden:\n{pad}")
                return
            
            # Lees de Excel-data in in het overzichtsbestand
            self.excel_data = pd.read_excel(pad)
            
            # Controleer of de kolom 'Participant' aanwezig is. Zo niet, geef een foutmelding en stop
            if 'Participant' not in self.excel_data.columns:
                messagebox.showerror("Fout", "Kolom 'Participant' ontbreekt.")
                return
            
            # Haal de unieke deelnemers/participanten op
            deelnemers = self.excel_data['Participant'].dropna().unique().astype(str)
            
            # Stel de waarden in van het dropdown-menu om een participant te selecteren
            self.participant_dropdown.configure(values=["Selecteer participant..."] + deelnemers.tolist())
            
            # Stel de standaard geselecteerde waar in in het dropdown-menu
            self.participant_dropdown.set("Selecteer participant...")
            
            # Maak checkboxen aan voor het selecteren van gewenste kolommen in de resultaten
            self._create_column_checkboxes()
            
            # Informeer de gebruiker dat het overzicht is geladen
            messagebox.showinfo("Klaar", "Overzicht geladen.")
        except Exception as e:
            # laat een foutmelding zien als er iets misgaat
            messagebox.showerror("Fout", f"Er is iets misgegaan:\n{e}")

    def _create_column_checkboxes(self):
        # Verwijder alle bestaande widgets in het kolommen-frae behalve het label
        for widget in self.columns_frame.winfo_children():
            if widget != self.columns_label:
                widget.destroy()

        # Definieer de standaard kolommen die aangevinkt moeten zijn
        standaard_kolommen = ["Participant", "Datum van meting", "Tijd meting begonnen", "Tijd meting beëindigd", "Slaapscore"]
        
        # Maak een lege dictionary om de checkbox-variabelen op te slaan
        self.column_checkboxes = {}
        
        # Maak voor elke kolom in de Excel-data een checkbox aan
        for col in self.excel_data.columns:
            # zet de checkbox standaad aan als de kolom in de standaardlijst staat
            var = tk.BooleanVar(value=(col in standaard_kolommen))
            
            # Maak een checkbox met de kolomnaam als tekst en koppel de variabele
            checkbox = ctk.CTkCheckBox(self.columns_frame, text=col, variable=var, command=self._update_textbox_with_selection)
            
            # Plaats de checkbox in het frame, links uitgelijnd met kleine verticale ruimte ertussen
            checkbox.pack(anchor="w", pady=2)
            
            # Sla de variabele op in de dictionairy
            self.column_checkboxes[col] = var

    def _update_textbox_with_selection(self):
        # Stop als er geen Excel-data is geladen
        if self.excel_data is None:
            return
        
        # Haal de geselecteerde participant uit de dropdown
        participant = self.participant_dropdown.get()
        
        # Als er geen participant geselcteerd is, geef instructie weer en updat status onderin
        if participant == "Selecteer participant..." or not participant:
            self.textbox.delete("1.0", "end")
            self.textbox.insert("end", "Selecteer een participant om data te tonen.")
            self.status_label.configure(text="Geen participant geselecteerd.")
            return
        
        # Maak lijst met kolommen die geselecteerd zijn via checkboxen
        geselecteerde_kolommen = [col for col, var in self.column_checkboxes.items() if var.get()]
        
        # Als er geen kolommen geselecteerd zijn, toon melding en update status onderin
        if not geselecteerde_kolommen:
            self.textbox.delete("1.0", "end")
            self.textbox.insert("end", "Geen kolommen geselecteerd.")
            self.status_label.configure(text="Geen kolommen geselecteerd.")
            return
        
        # Filter data voor de geselecteerde participant
        data = self.excel_data[self.excel_data['Participant'].astype(str) == participant]
        
        # Als er geen data is voor deze participant, toon melding en update status
        if data.empty:
            self.textbox.delete("1.0", "end")
            self.textbox.insert("end", "Geen data gevonden voor deze participant.")
            self.status_label.configure(text="Geen data voor participant gevonden.")
            return
        
        # Maak subset van de geselecteerde kolommen, rest index voor nette tabel
        subset = data[geselecteerde_kolommen].reset_index(drop=True)
        
        # Format data als een nette tabel met grid-stijl
        formatted = tabulate(subset, headers="keys", tablefmt="grid", showindex=False)
        
        # Maak tekstvak leeg en vul de geformatteerde tabel
        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", formatted)
        
        # Update status onderaan met welke participant getoond wordt 
        self.status_label.configure(text=f"Gegevens voor participant '{participant}' getoond.")

    # Roept de functie aan om het tekstvak bij te werken met de data van de geselecteerde participant
    def show_participant_data(self, participant):
        self._update_textbox_with_selection()

    # Laat een dialoog zien om een Excel-bestand te selecteren met de slaapscore-matrix
    def load_sleep_matrix(self):
        bestand = filedialog.askopenfilename(title="Selecteer Slaapscore_matrix.xlsx", filetypes=[("Excel-bestanden", "*.xlsx")])
        
        # als er geen bestand is geselecteerd, stop dan
        if not bestand:
            return
        try:
            # lees het geselecteerde Excel-bestand in als DataFrame
            self.matrix_df = pd.read_excel(bestand)
            
            # Zet de inhoud van het bestand om in een mooie tabelvorm (grid) en toon dit in de tekstbox
            tekst = tabulate(self.matrix_df, headers="keys", tablefmt="grid", showindex=False)
            self.matrix_textbox.delete("1.0", "end")
            self.matrix_textbox.insert("end", tekst)

            # Zoek kolommen die beginnen met "Nacht van" om als datums te gebruiken
            datumkolommen = [col for col in self.matrix_df.columns if col.startswith("Nacht van")]
            
            # Vul het dropdown-menu met de gevonden datums
            self.date_dropdown.configure(values=datumkolommen)
            
            if datumkolommen:
                # Selecteer standaard de eerste datum en toon direct de scores ervan
                self.date_dropdown.set(datumkolommen[0])
                self.show_scores_by_date(datumkolommen[0])
            else:
                # Als er geen datums zijn gevonden, geef dit aan in het dropdown-menu
                self.date_dropdown.set("Geen datums gevonden")
        
        # Als het laden mislukt, toon dan een foutmelding
        except Exception as e:
            messagebox.showerror("Fout", f"Kan bestand niet laden:\n{e}")

    # Toon de slaapscores van deelnemers voor de geselecteerde nacht
    def show_scores_by_date(self, geselecteerde_datum):
        
        # Controleer of het matrix-dataframe bestaat (is geladen)
        if not hasattr(self, "matrix_df"):
            return
        
        # Controleer of de geselecteerde datumkolom aanwezig is in de data
        if geselecteerde_datum not in self.matrix_df.columns:
            self.matrix_textbox.delete("1.0", "end")
            self.matrix_textbox.insert("end", "Datum niet gevonden in matrix.")
            return
        
        # Maak een kopie van het DataFrame met alleen de kolommen 'Participant en de geselecteerde datum
        df_subset = self.matrix_df[["Participant", geselecteerde_datum]].copy()
        
        # Filter rijen waar NaN staat in de geselecteerde datumkolom
        df_subset = df_subset[df_subset[geselecteerde_datum].notna()]
        
        # Zet de gefilterde data om naar een net opgemaakte tabel en toon dit in de tekstbox
        tekst = tabulate(df_subset, headers="keys", tablefmt="grid", showindex=False)
        self.matrix_textbox.delete("1.0", "end")
        self.matrix_textbox.insert("end", tekst)



def start_GUI():
    app = App()
    # Start de hoofdloop van de GUI. Dit houdt het venster actief tot dat het gesloten wordt
    app.mainloop()

# Start de applicatie alleen als dit script direct wordt uitgevoerd
if __name__ == "__main__":
    start_GUI()