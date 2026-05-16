import os
import requests
import json
import time
from datetime import datetime



class ClinicalDataExtractor:

    '''
    LLM component of the Operating Room Charting Assistant (ORCA) responsible for:
        interpreting raw ASR surgical transcripts,
        extracting clinical data for specified fields,
        and returning extracted data in JSON format according to provided template.

    Set Up
    ------
        Run: pip install ollama.
        Run: brew services start ollama.
        Run: ollama pull "model name." Default: "llama3.1:8b".
        Initialize ClinicalDataExtractor with model name.

    Methods
    -------
        create_extraction_prompt : creates prompt for clinical data extraction.
        extract : extracts clinical data from a transcript and saves
            the extracted data into a provided JSON report template.
    '''

    
    def __init__(self, cde_folder_path: str = 'reports', 
                 cde_model = 'llama3.1:8b', 
                 base_url = 'http://localhost:11434'):
        
        '''
        initializes the extractor and verifies Ollama is running with the requested model.

        Parameters
        ----------
            cde_folder_path : str
                folder to store generated reports. Default: "reports".
            cde_model : str
                model name used in pull. Default: "llama3.1:8b".
            base_url : str
                Ollama API endpoint. Default: 'http://localhost:11434'.
        '''

        #initialized class variables...
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_path = f'{cde_folder_path}/extracted_report_{timestamp}.json'
        os.makedirs(cde_folder_path, exist_ok = True)
        self.model = cde_model
        self.api_url = f'{base_url}/api/generate'
        self.empty_responses = {'none', 'n/a', 'unknown', '<none>'}
        
        
        #check if ollama is running...
        try:

            #ask ollama for downloaded models '/api/tags'; give up after 5 seconds...
            response = requests.get(f'{base_url}/api/tags', timeout = 5)

            #200 = everything worked, so navigate to model names...
            if response.status_code == 200:
                try:
                    models = response.json().get('models', [])
                except json.JSONDecodeError:
                    raise RuntimeError('🅇 Error: Ollama returned an unreadable response.')
                model_names = [m['name'] for m in models]
                
                #if initialized model name not downloaded, throw an error warning...
                if self.model not in model_names:
                    available = ', '.join(model_names) if model_names else 'None'
                    msg = (
                        f'🅇 Error: Model "{self.model}" not found.\n'
                        f'Available models: {available}\n'
                        f'To install: ollama pull {self.model}'
                    )
                    raise RuntimeError(msg)
                else:
                    print(f'\n✔ Connected to Ollama - using {self.model}\n')
            
            #if request timed out, throw an error...
            else:
                raise RuntimeError('🅇 Error: Ollama is not responding properly.')

        #if base url is wrong or timed out, throw an error warning...
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            msg = (
                '🅇 Error: Cannot connect to Ollama.\n'
                'How to fix:\n'
                '  1. Run: brew services start ollama\n'
                '  2. Run: ollama serve\n'
                '  3. Verify base_url with port number.'
            )
            raise RuntimeError(msg) from e


    def create_extraction_prompt(self, field_description: str) -> str:

        '''
        creates a structured prompt for clinical data extraction.

        Parameters
        ----------
            field_description : str
                field prompting question to embed in the prompt.

        Returns
        -------
            extraction_prompt : str
                formatted prompt string containing the transcript chunk,
                field description, and output instructions.
        '''

        #build prompt...
        extraction_prompt = (
            f'Transcript: {self.transcript_chunk}\n\n'
            f'{field_description}\n'
            f'Answer in 15 words or less. If not mentioned, write <none>.\n'
            f'Answer:'
        )
        return extraction_prompt


    def _init_section(self, d: dict) -> dict:

        '''
        recursively initializes all leaf fields in a template section to ['<none>'].

        Parameters
        ----------
            d : dict
                a section sub-dict from the JSON template.

        Returns
        -------
            result : dict
                same structure with every leaf value set to ['<none>'].
        '''

        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                if 'description' in value:
                    result[key] = ['<none>']
                else:
                    result[key] = self._init_section(value)
        return result


    def _initialize_report(self, template: dict) -> dict:

        '''
        builds a blank report dict from the full template, with all leaf values set to ['<none>'].

        Parameters
        ----------
            template : dict
                full JSON template loaded from disk.

        Returns
        -------
            report : dict
                initialized report with every field set to ['<none>'].
        '''

        report = {
            sec_key: self._init_section(sec_val)
            for sec_key, sec_val in template.items()
            if isinstance(sec_val, dict)
        }
        return report


    def extract_section(self, section_key: str, section_dict: dict,
                        temperature: float, show_output: bool) -> None:

        '''
        recursively collects all leaf fields with descriptions, calls the LLM for
        each, then writes results back to the report file.

        Parameters
        ----------
            section_key : str
                top-level key name, e.g. "patient_info".
            section_dict : dict
                the dict under that key.
            temperature : float
                0.05 by default (lower = more deterministic).
            show_output : bool
                print extracted fields to console if True.
        '''

        #collect all leaf fields with descriptions in this section...
        def collect_fields(d: dict, path: str = '') -> dict:
            fields = {}
            for key, value in d.items():
                current_path = f'{path}.{key}' if path else key
                if isinstance(value, dict):
                    if 'description' in value:
                        fields[current_path] = value
                    else:
                        fields.update(collect_fields(value, current_path))
            return fields

        #gather all fields for this section...
        batch_fields = collect_fields(section_dict)

        if not batch_fields:
            return

        #load existing report from disk...
        start = time.time()
        if os.path.exists(self.report_path):
            with open(self.report_path, 'r', encoding = 'utf-8') as f:
                report = json.load(f)
        else:
            report = {}
        
        for field_path, field_value in batch_fields.items():

            try:
                #create prompt for this individual field...
                payload = {
                    'model': self.model,
                    'prompt': self.create_extraction_prompt(field_value['description']),
                    'stream': False,
                    'temperature': temperature,
                    'options': {'num_predict': 64}
                }

                response = requests.post(self.api_url, json = payload, timeout = 30)
                response.raise_for_status()
                result = response.json()
                raw_text = result.get('response', '')

                #normalize generated text...
                generated_text = str(raw_text).strip()
                generated_lower = generated_text.lower()

                #skip empty or invalid responses (exact match against sentinels)...
                if not generated_lower or generated_lower in self.empty_responses:
                    if show_output:
                        print(f'{field_path} skipped ┈┈→ {generated_text}')
                    continue

                #prepend section key to path to preserve nesting...
                full_path = f'{section_key}.{field_path}'
                keys = full_path.split('.')
                node = report
                for key in keys[:-1]:
                    node = node.setdefault(key, {})

                #replace <none> placeholder, then append if not duplicate...
                final_key = keys[-1]
                if final_key in node and isinstance(node[final_key], list):
                    existing = [v for v in node[final_key] if v.lower() != '<none>']
                    if generated_text not in existing:
                        existing.append(generated_text)
                    node[final_key] = existing if existing else [generated_text]
                else:
                    node[final_key] = [generated_text]

                #display output if requested...
                if show_output:
                    print(f'{final_key} ┈┈→ {generated_text}')

                #write updated report back to disk after each field...
                with open(self.report_path, 'w', encoding = 'utf-8') as f:
                    json.dump(report, f, indent = 4)

            except requests.exceptions.RequestException as e:
                print(f'🅇 Error on field "{field_path}": {e} — skipping.')
                continue

        if show_output:
            print(f'{section_key} extracted! Processing time: {time.time() - start:.3f} seconds.\n')


    def extract(self, transcript_chunk: str, template_path: str,
                temperature: float = 0.05, show_output: bool = True) -> None:

        '''
        extracts clinical data from a transcript and saves
        the extracted data into a JSON file.

        Parameters
        ----------
            transcript_chunk : str
                raw ASR surgical transcript.
            template_path : str
                local path to JSON template with given fields and descriptions.
            temperature : float
                0.05 by default (lower = more deterministic).
            show_output : bool
                print extracted fields to console if True.
        '''

        #load json from path...
        with open(template_path, 'r', encoding = 'utf-8') as f:
            self.template = json.load(f)

        #store transcript chunk for use in prompt creation...
        self.transcript_chunk = transcript_chunk

        #pre-initialize report with all template fields set to ['<none>']...
        initial_report = self._initialize_report(self.template)
        with open(self.report_path, 'w', encoding = 'utf-8') as f:
            json.dump(initial_report, f, indent=4)

        #extract each top-level section with one llm call per field...
        tot_start = time.time()
        for section_key, section_value in self.template.items():
            if isinstance(section_value, dict):
                self.extract_section(section_key, section_value, temperature, show_output)

        elapsed = time.time() - tot_start
        print(f'\n✔ Report saved to: {self.report_path}')
        print(f'Extraction processing time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s.\n')