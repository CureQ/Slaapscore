# Bestandsnaam: Overzicht_movesense.py
# Geschreven door: Esmee Springer
# Voor het laatst bewerkt op: 22-05-2025

## Deze code is bedoeld voor het maken van een overzicht van de resultaten van de MoveSense. 

# Importeren van de benodigde pakkages
import os
import re
import pandas as pd
from datetime import datetime, time
from datetime import timedelta

# Functie die ervoor zorgt dat alle Excel output worden gegenereerd.
# Neemt een map (root_dir) met Excel-bestanden als input en verwerkt alle data daarin
# Verzamelt, filtert en berekent slaapgegevens, en slaat resultaten op in verschillende Excel-bestanden.
def run_movesense_analysis(root_dir):

    # Functie om een specifieke waarde op te halen uit de eerste kolom van een DataFrame
    def get_value(df, zoektekst):
        # Zoek naar de rij waar de eerste kolom gelijk is aan 'zoektekst' en retourneer de waarde uit de tweede kolom.
        match = df[df.iloc[:, 0] == zoektekst]
        if not match.empty:
            return match.iloc[0, 1]
        return None

    # Lijst om later de verzamelde rijen met data in op te slaan
    rows = []

    # Loop door alle bestanden in de opgegeven map 'root_dir'
    for filename in os.listdir(root_dir):
        # Verwerk alleen Excel-bestanden (.xlsx)
        if filename.endswith(".xlsx"):
            file_path = os.path.join(root_dir, filename)
            try:
                # Probeer het Excel-bestand te openen
                xls = pd.ExcelFile(file_path)
            except Exception as e:
                # Bij een fout bij het openen, geeft een melding en sla het bestand over
                print(f"Fout bij openen van bestand {filename}: {e}")
                continue
            
            # Haal het eerste getal uit de bestandsnaam om de deelnemer te identificeren
            participant_name = re.findall(r'\d+', filename)[0]
            
            # Loop door alle sheets in het Excel_bestand
            for sheet in xls.sheet_names:
                # Lees de sheet in als dataframe zonder header (header=None)
                df = xls.parse(sheet, header=None)

                # Haal verschillende belangrijke metingen op uit het dataframe via 'get_value'
                datum = get_value(df, "Datum van meting")
                tijd_begonnen = get_value(df, "Tijd meting begonnen")
                tijd_slaap = get_value(df, "Tijd in slaap gevallen")
                tijd_wakker = get_value(df, "Tijd wakker geworden")
                tijd_einde = get_value(df, "Tijd meting beëindigt")
                wakker_voor_in_slaap = get_value(df, "Minuten wakker voordat deelnemer in slaap viel")
                aantal_keer_wakker = get_value(df, "Aantal keer wakker geworden per nacht")

                totaal_minuten_gemeten = get_value(df, "Totaal aantal minuten gemeten")
                minuten_W = get_value(df, "Totaal aantal minuten in wakker fase")
                minuten_N1 = get_value(df, "Totaal aantal minuten in N1 fase")
                minuten_N2 = get_value(df, "Totaal aantal minuten in N2 fase")
                minuten_N3 = get_value(df, "Totaal aantal minuten in N3 fase")
                minuten_REM = get_value(df, "Totaal aantal minuten in REM fase")

                totaal_uren_gemeten = get_value(df, "Totaal aantal uren gemeten")
                totaal_uren_geslapen = get_value(df, "Totaal aantal uren geslapen")

                # Bereken slaap efficiëntie(%) als verhouding van geslapen uren ten opzichte van totaal gemeten uren
                slaap_efficiëntie = None
                try:
                    if totaal_uren_gemeten and totaal_uren_geslapen:
                        slaap_efficiëntie = (float(totaal_uren_geslapen) / float(totaal_uren_gemeten)) * 100
                except:
                    slaap_efficiëntie = None

                # Als slaap efficiëntie niet berekend kan worden, zet deze dan op 0
                if slaap_efficiëntie is None:
                    slaap_efficiëntie = 0

                # Stel de percentages voor verschillende slaapfasen in
                perc_W = perc_N1 = perc_N2 = perc_N3 = perc_REM = None
                try:
                    totaal_minuten_gemeten = float(totaal_minuten_gemeten)
                    totaal_minuten_geslapen = float(totaal_uren_geslapen) * 60 if totaal_uren_geslapen else 0
                    if totaal_minuten_gemeten > 0:
                        # Bereken het percentage wakker ten opzichte van het totaal aantal minuten gemeten
                        perc_W = (float(minuten_W) / totaal_minuten_gemeten) * 100 if minuten_W else 0
                        # Bereken de percentages in ieder slaapfasen ten opzichte van het totaal aantal minuten geslapen
                        perc_N1 = (float(minuten_N1) / totaal_minuten_geslapen) * 100 if minuten_N1 else 0
                        perc_N2 = (float(minuten_N2) / totaal_minuten_geslapen) * 100 if minuten_N2 else 0
                        perc_N3 = (float(minuten_N3) / totaal_minuten_geslapen) * 100 if minuten_N3 else 0
                        perc_REM = (float(minuten_REM) / totaal_minuten_geslapen) * 100 if minuten_REM else 0
                # Bij eventuele fouten in de berekening hierboven wordt de fout genegeerd, zodat de code verder runt.
                except:
                    pass

                # Voeg de verwerkte gegevens van deze sheet toe aan de lijst van alle rijen
                rows.append([
                    participant_name, datum, tijd_begonnen, tijd_slaap, tijd_wakker, tijd_einde, wakker_voor_in_slaap,
                    aantal_keer_wakker, perc_W, perc_N1, perc_N2, perc_N3, perc_REM,
                    totaal_uren_gemeten, totaal_uren_geslapen, slaap_efficiëntie
                ])

    # Definieer de kolomnamen voor het uiteindelijke DataFrame
    columns = [
        "Participant", "Datum van meting", "Tijd meting begonnen", "Tijd in slaap gevallen", "Tijd wakker geworden",
        "Tijd meting beëindigd", "Inslaaptijd (min)", "Aantal keer wakker geworden per nacht",
        "% in W", "% in N1", "% in N2", "% in N3", "% in REM",
        "Totaal aantal uren gemeten", "Totaal aantal uren geslapen", "Slaap efficiëntie"
    ]
    # Maak een DataFrame van alle verzamelde rijen en geef deze de kolomnamen
    output_df = pd.DataFrame(rows, columns=columns)
    # Converteer de kolom 'Participant' naar numeriek. Waarden die niet kunnen worden geconverteerd worden NaN
    output_df['Participant'] = pd.to_numeric(output_df['Participant'], errors='coerce')

    # de functie parse_time probeert een tijdwaarde te converteren naar een datetime.time object
    def parse_time(tijd):
        # Converteer een tijd string naar een datetime.time object
        # Probeer eerst naar format (%H:%M:%S), anders naar (%H:%M)
        # Als de waarde ontbreekt of niet te converteren is, restourneer None

        if pd.isna(tijd):
            return None
        try:
            return datetime.strptime(str(tijd), "%H:%M:%S").time()
        except:
            try:
                return datetime.strptime(str(tijd), "%H:%M").time()
            except:
                return None

    # De functie is_valid_row controleert of een rij geldig is op basis van bepaalde criteria
    def is_valid_row(row):
        start = parse_time(row["Tijd meting begonnen"]) # Parse starttijd naar datetime.time
        end = parse_time(row["Tijd meting beëindigd"]) # Parse eindtijd naar datetime.time
        duur = row["Totaal aantal uren gemeten"] # Totale duur van de meting in uren
        slaaptijd = row["Totaal aantal uren geslapen"] # Totale slaaptijd in uren

        # Controleer of starttijd, eindtijd, duur en slaaptijd geldig en aanwezig zijn
        if start is None or end is None or pd.isna(duur) or pd.isna(slaaptijd):
            return False

        # Check of de starttijd tussen 09:00 en 01:00 ligt 
        in_start_range = (time(9, 0) <= start or start <= time(1, 0))
        # Check of eindtijd tussen 05:00 en 10:00 ligt
        in_end_range = time(5, 0) <= end <= time(10, 0)
        # # Check of de duur van de meting minimaal 5 uur is
        long_enough = float(duur) >= 5
        # Check of de slaaptijd van de participant minimaal 3 uur is
        sleep_enough = float(slaaptijd) >= 3

        # Alleen teruggeven wanneer er aan alle 4 bovengenoemde eisen wordt voldaan
        return in_start_range and in_end_range and long_enough and sleep_enough

    # Eerste opslag van de volledige ruwe tabel naar een Excel-bestand
    # Bestandspad voor het opslaan van het Excel-bestand
    output_path = r"C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/Overzicht_resultaten.xlsx"
    # DataFrame opslaan als Excel zonder index
    output_df.to_excel(output_path, index=False)
    # Geeft melding wanneer het overzicht is opgeslagen
    print(f"Gereed! Overzicht opgeslagen op: {output_path}")

    # Filter het DataFrame volgens de functie is_valid_row
    filtered_df = output_df[output_df.apply(is_valid_row, axis=1)]

    def lineaire_punten(waarde, ideaal_min, ideaal_max, min_grens, max_grens, max_punten):
        # Return 0 als de waarde ontbreekt
        if waarde is None:
            return 0
        # Converteer waarde naar een float voor de berekeningen
        waarde = float(waarde)

        # Geef maximale punten als waarde binnen ideaal bereik valt
        if ideaal_min <= waarde <= ideaal_max:
            return max_punten
        
        # Lineaire afname van punten als waarde onder ideaal_min maar boven min_grens valt
        elif waarde < ideaal_min and waarde >= min_grens:
            return max_punten * (waarde - min_grens) / (ideaal_min - min_grens)
        
        # Lineaire afname van punten als waarde boven ideaal_max maar onder max_grens valt
        elif waarde > ideaal_max and waarde <= max_grens:
            return max_punten * (max_grens - waarde) / (max_grens - ideaal_max)
        
        # Geen punten als waarde buiten alle grenzen valt
        else:
            return 0

    def bereken_slaapscore_v2(rij):
        # Probeer totale tijd geslapen en totale tijd gemeten om te zetten naar minuten
        # return None als 0 minuten geslapen of in bed
        try:
            geslapen_min = float(rij["Totaal aantal uren geslapen"]) * 60
            in_bed_min = float(rij["Totaal aantal uren gemeten"]) * 60
            if geslapen_min == 0 or in_bed_min == 0:
                return None

            # Haal de percentages in iedere slaapfasen op, vervang NaN door 0
            perc_W = float(rij["% in W"]) if not pd.isna(rij["% in W"]) else 0
            perc_N1 = float(rij["% in N1"]) if not pd.isna(rij["% in N1"]) else 0
            perc_N2 = float(rij["% in N2"]) if not pd.isna(rij["% in N2"]) else 0
            perc_N3 = float(rij["% in N3"]) if not pd.isna(rij["% in N3"]) else 0
            perc_REM = float(rij["% in REM"]) if not pd.isna(rij["% in REM"]) else 0

            # Haal inslaaptijd en aantal keer wakker geworden op, vervang NaN door 999 (als 'slechte' score)
            inslaaptijd = float(rij["Inslaaptijd (min)"]) if not pd.isna(rij["Inslaaptijd (min)"]) else 999
            aantal_wakker = float(rij["Aantal keer wakker geworden per nacht"]) if not pd.isna(rij["Aantal keer wakker geworden per nacht"]) else 999

            # Stel perc_wakker gelijk aan perc_W
            perc_wakker = perc_W
            
            # Bereken totaal score op basis van lineaire punten per meting
            score = 0
            score += lineaire_punten(geslapen_min, 420, 540, 300, 660, 30)
            score += lineaire_punten(perc_wakker, 5, 10, 0, 15, 10)
            score += lineaire_punten(perc_N1, 2, 5, 0, 8, 5)
            score += lineaire_punten(perc_N2, 45, 55, 35, 65, 15)
            score += lineaire_punten(perc_N3, 13, 23, 3, 33, 5)
            score += lineaire_punten(perc_REM, 20, 25, 15, 30, 15)
            score += lineaire_punten(inslaaptijd, 0, 20, 0, 45, 10)
            score += lineaire_punten(aantal_wakker, 0, 4, 0, 7, 10)

            # Zorg dat de score maximaal 100 is en rond af op 1 decimaal
            return round(min(score, 100), 1)
        except:
            return None

    # Voeg de kolom Slaapscore toe aan het gefilterde DataFrame
    filtered_df["Slaapscore"] = filtered_df.apply(bereken_slaapscore_v2, axis=1)

    # Tweede opslag van gefilterde dataset inclusief slaapscore
    filtered_output_path = r"C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/Overzicht_resultaten_gefilterd.xlsx"
    # Schrijf de gefilterde DataFrame naar een Excel-bestand
    filtered_df.to_excel(filtered_output_path, index=False)
    # Melding voor het succesvol opslaan van het overzicht met de slaapscore
    print("Gefilterd overzicht met slaapscore opgeslagen op:", filtered_output_path)

    # Functie om de daadwerkelijke nacht van de meting te bepalen
    def bepaal_daadwerkelijke_nacht(rij):
        # Haal de datum en starttijd van de meting op
        datum = rij["Datum van meting"]
        tijd_begonnen = parse_time(rij["Tijd meting begonnen"])

        # Controleer of er ontbrekende waarde zijn
        if pd.isna(datum) or tijd_begonnen is None:
            return None

        try:
            # Zet de datum om naar een datetime-object
            datum_obj = pd.to_datetime(datum, dayfirst=False)  # Datum van meting wordt datetime
            
            # Indien vóór 12:00 (bijv. 01:00), reken het tot de vorige nacht
            if tijd_begonnen < time(12, 0):  
                datum_obj = datum_obj - timedelta(days=1)
            
            # retourneer de gecorrigeerde datum als 'daadwerkelijke nacht'
            return datum_obj.date()
             
        except:
            return None


    # Voeg de kolom 'Daadwerkelijke nacht' toe aan de gefilterde dataset
    filtered_df["Daadwerkelijke nacht"] = filtered_df.apply(bepaal_daadwerkelijke_nacht, axis=1)

    # Pad waar de excel output wordt opgeslagen
    output_path_daadwerkelijke_nacht = r"C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/Overzicht_resultaten_met_nacht.xlsx"
    # Opslaan van de gefilterde dataset met de toegevoegde kolom 'Daadwerkelijke nacht'
    filtered_df.to_excel(output_path_daadwerkelijke_nacht, index=False)
    print("Toegevoegde kolom 'Daadwerkelijke nacht' opgeslagen in:", output_path_daadwerkelijke_nacht)

    # Herhaald opslaan van dezelfde gefilterde dataset, nu met slaapscore
    filtered_df.to_excel(filtered_output_path, index=False)
    print("Gefilterd overzicht met slaapscore opgeslagen op:", filtered_output_path)

    # === HIER START het maken van de slaapscore matrix===

    # Maak een kopie om verdere bewerkingen te doen zonder de originele gefilterde dataset aan te passen
    df_met_nacht = filtered_df.copy()

    # Zet de kolom 'Daadwerkelijke nacht' om naar een datetime-object
    df_met_nacht["Daadwerkelijke nacht"] = pd.to_datetime(df_met_nacht["Daadwerkelijke nacht"], errors="coerce")
    # Zorg dat 'Daadwerkelijke nacht' als stringdatum wordt opgeslagen voor de kolomnamen, bijvoorbeeld 'Nacht van 21-05-2025
    df_met_nacht["Nacht van"] = df_met_nacht["Daadwerkelijke nacht"].dt.strftime("Nacht van %Y-%m-%d")

    # Maak een matrix met de deelnemers als rijen en de nachten als kolommen, gevuld met de slaapscores
    slaapscore_matrix = df_met_nacht.pivot_table(
        index="Participant",
        columns="Nacht van",
        values="Slaapscore",
        aggfunc="first"  # of 'mean' indien meerdere metingen/nacht
    )

    # Verwijder de naam van de kolom-indez en sorteer de kolommen op de datum
    slaapscore_matrix.columns.name = None
    slaapscore_matrix = slaapscore_matrix.sort_index(axis=1)

    # Exporteer de matrix naar een naar Excel-bestand
    matrix_output_path = r"C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/Slaapscore_matrix.xlsx"
    slaapscore_matrix.to_excel(matrix_output_path)

    print(f"Slaapscore matrix opgeslagen op: {matrix_output_path}")


# Startpunt van het script
# Geef hier het pad naar de map met alle Excelbestanden op (Movesense_data_resultaten)
if __name__ == "__main__":
    root_dir = r"C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/MoveSense_data_resultaten"
    run_movesense_analysis(root_dir)
